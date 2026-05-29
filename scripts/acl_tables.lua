-- Pandoc Lua filter: convert all Table elements to a 2-column-spanning
-- ACL-compatible \begin{table*}\begin{tabular}...\end{tabular}\end{table*}
-- block, strip leading "N.M.K " prefixes from headings (so LaTeX section
-- numbering doesn't collide with markdown's manual numbers), and rewrite
-- long inline Code spans (>= 18 chars) as \seqsplit{\texttt{...}} so they
-- can break across the narrow ACL column. In heading context, plain
-- \texttt{} is used (\seqsplit is fragile in moving args).
--
-- Usage: pandoc --lua-filter=acl_tables.lua ...

local function escape_for_texttt(s)
  s = s:gsub('\\', '\\textbackslash{}')
  s = s:gsub('([_#%%&{}$])', '\\%1')
  return s
end

-- For long monospace identifiers (file paths, env vars, brace-expansions),
-- insert *discouraged* break points at structural boundaries: \penalty 5000
-- means "break here only if absolutely necessary, otherwise prefer to stretch
-- the line slightly". Without these, \seqsplit either fires per-character
-- (ugly) or doesn't fire at all (overfull). With unconditional \allowbreak,
-- TeX takes every opportunity and produces severely stretched lines elsewhere
-- (badness 10000 underfull hboxes). \penalty 5000 is the Goldilocks setting.
local function insert_breakpoints(s)
  -- After underscore-escape: \_ → \_\penalty5000{}
  s = s:gsub('\\_', '\\_\\penalty5000{}')
  -- After comma: prefer breaking at commas in brace-expansions
  s = s:gsub(',', ',\\penalty5000{}')
  -- After escaped open brace: \{ → \{\penalty5000{}
  s = s:gsub('\\{', '\\{\\penalty5000{}')
  -- Before escaped close-brace: \penalty5000{}\}
  s = s:gsub('\\}', '\\penalty5000{}\\}')
  -- After slash (lower penalty — more natural break point in URLs/paths)
  s = s:gsub('/', '/\\penalty3000{}')
  -- After colon: arXiv:1234.5678, URLs (low penalty for arXiv IDs)
  s = s:gsub(':', ':\\penalty3000{}')
  -- After hyphen: TODO-RESEARCH.md, kebab-case identifiers
  s = s:gsub('%-', '-\\penalty3000{}')
  -- Before period (only if followed by extension-like chars): foo.md, foo.json
  s = s:gsub('%.', '.\\penalty3000{}')
  return s
end

local function code_to_latex(c, in_header)
  if in_header then
    return pandoc.RawInline('latex', '\\texttt{' .. escape_for_texttt(c.text) .. '}')
  end
  if #c.text >= 12 then
    return pandoc.RawInline('latex',
      '\\seqsplit{\\texttt{' .. insert_breakpoints(escape_for_texttt(c.text)) .. '}}')
  end
  return c
end

function Header(h)
  -- Strip a leading "<number>(.<number>)* " prefix so LaTeX's auto-numbering
  -- doesn't collide with the markdown manual numbers.
  if #h.content > 0 and h.content[1].t == 'Str' then
    local first = h.content[1].text
    local m = first:match('^([%dA]+%.[%w%.]*)$') or first:match('^(%d+%.?)$')
    if m then
      table.remove(h.content, 1)
      if #h.content > 0 and h.content[1].t == 'Space' then
        table.remove(h.content, 1)
      end
    end
  end
  -- Walk the header inlines and rewrite Code with in_header=true so we get
  -- plain \texttt{} (no \seqsplit) inside section titles.
  for i, inline in ipairs(h.content) do
    if inline.t == 'Code' then
      h.content[i] = code_to_latex(inline, true)
    end
  end
  return h
end

function Code(c)
  -- Body-level Code (Header filter handled in-heading cases already).
  return code_to_latex(c, false)
end

function Table(t)
  -- Reshape using pandoc's writer to LaTeX with table style overridden.
  -- pandoc.write doesn't expose a knob for "no longtable", so we emit raw LaTeX
  -- by walking t.bodies, t.head, t.foot and rendering cells.
  local function render_inlines(inlines)
    return pandoc.write(pandoc.Pandoc({pandoc.Plain(inlines)}), 'latex')
  end
  local function render_blocks(blocks)
    return pandoc.write(pandoc.Pandoc(blocks), 'latex')
  end

  -- Determine column count from t.colspecs
  local ncols = #t.colspecs
  -- Default to all-left; could honor colspecs[i][1] if needed
  local colspec_chars = {}
  for i = 1, ncols do
    local align = t.colspecs[i][1]
    if align == 'AlignRight' then
      colspec_chars[i] = 'r'
    elseif align == 'AlignCenter' then
      colspec_chars[i] = 'c'
    else
      colspec_chars[i] = 'l'
    end
  end
  local colspec = '@{}' .. table.concat(colspec_chars, '') .. '@{}'

  -- Render rows: header rows + body rows
  local function row_to_latex(row)
    local cells = {}
    for _, cell in ipairs(row.cells) do
      cells[#cells+1] = render_blocks(cell.contents):gsub('\n+$', ''):gsub('^%s+', '')
    end
    return table.concat(cells, ' & ') .. ' \\\\'
  end

  local out = {}
  table.insert(out, '\\begin{table*}[t]')
  table.insert(out, '\\centering')
  -- Only shrink-to-fit, never upscale. Narrow tables stay at natural size
  -- (else \resizebox blows them up to \textwidth and the font looks huge).
  table.insert(out, '\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%')
  table.insert(out, '\\begin{tabular}{' .. colspec .. '}')
  table.insert(out, '\\toprule')
  for _, row in ipairs(t.head.rows) do
    table.insert(out, row_to_latex(row))
  end
  table.insert(out, '\\midrule')
  for _, body in ipairs(t.bodies) do
    for _, row in ipairs(body.body) do
      table.insert(out, row_to_latex(row))
    end
  end
  table.insert(out, '\\bottomrule')
  table.insert(out, '\\end{tabular}')
  table.insert(out, '}')  -- close \resizebox
  table.insert(out, '\\end{table*}')

  return pandoc.RawBlock('latex', table.concat(out, '\n'))
end
