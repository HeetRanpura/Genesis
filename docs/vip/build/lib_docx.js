// Shared look and helpers for the VIP reports (docx). Used by build_sem4_reports.js and build_sem5_reports.js.
const fs = require("fs");
const path = require("path");
const sharp = require("sharp");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, HeadingLevel, BorderStyle, Footer, Header, PageNumber, LevelFormat, PageBreak, TabStopType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..", "..", "..");
const OUT = path.resolve(__dirname, "..");
const INK = "1B1B1A", INK2 = "52514E", MUTED = "8A8984", ACCENT = "1F4E8C", RULE = "C9C8C3", TINT = "EEF3FA";
const CONTENT_DXA = 9026;            // A4 width 11906 minus two 1" margins
const MAX_IMG_PX = 600;              // content width at 96 dpi

// "**bold**" and "`code`" inline markup -> TextRuns
function runs(text, base0 = {}) {
  const base = { font: "Arial", size: 21, ...base0 };   // explicit on every run: Pages ignores document defaults
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ ...base, text: t.slice(2, -2), bold: true }));
    else out.push(new TextRun({ ...base, text: t.slice(1, -1), font: "Courier New", size: base.size - 2 }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

const P = (text, opts = {}) => new Paragraph({ style: "Normal", children: runs(text, opts.run), spacing: { after: 120, line: 288 },
  alignment: opts.align, ...opts.para });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, keepNext: true, children: [new TextRun({ text: t, font: "Arial", size: 32, bold: true, color: ACCENT })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true, children: [new TextRun({ text: t, font: "Arial", size: 25, bold: true, color: INK })] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, keepNext: true, children: [new TextRun({ text: t, font: "Arial", size: 22, bold: true, color: INK2 })] });
const bullets = (items, level = 0) => items.map((t) => new Paragraph({ style: "Normal", numbering: { reference: "bullets", level },
  children: runs(t), spacing: { after: 60, line: 276 } }));
const numbered = (items, ref = "numbers") => items.map((t) => new Paragraph({ style: "Normal", numbering: { reference: ref, level: 0 },
  children: runs(t), spacing: { after: 60, line: 276 } }));
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

function callout(title, lines) {
  const border = { style: BorderStyle.SINGLE, size: 4, color: "B9CCE6" };
  return new Table({
    width: { size: CONTENT_DXA, type: WidthType.DXA }, columnWidths: [CONTENT_DXA],
    rows: [new TableRow({ children: [new TableCell({
      width: { size: CONTENT_DXA, type: WidthType.DXA }, shading: { type: ShadingType.CLEAR, fill: TINT, color: "auto" },
      borders: { top: border, bottom: border, left: border, right: border },
      margins: { top: 120, bottom: 120, left: 180, right: 180 },
      children: [new Paragraph({ style: "Normal", children: [new TextRun({ text: title, bold: true, color: ACCENT, font: "Arial", size: 21 })], spacing: { after: 80 } }),
        ...lines.map((l) => new Paragraph({ style: "Normal", children: runs(l), spacing: { after: 60, line: 276 } }))],
    })] })],
  });
}

// widths: fractions summing to 1
function table(headers, rows, widths, opts = {}) {
  const cols = widths.map((w) => Math.round(w * CONTENT_DXA));
  cols[cols.length - 1] = CONTENT_DXA - cols.slice(0, -1).reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: RULE };
  const borders = { top: border, bottom: border, left: border, right: border };
  const fs = opts.size || 18;
  const cell = (t, i, head) => new TableCell({
    width: { size: cols[i], type: WidthType.DXA }, borders,
    shading: head ? { type: ShadingType.CLEAR, fill: "E3EAF4", color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: String(t).split("\n").map((line) => new Paragraph({ style: "Normal", children: runs(line, { size: fs, bold: head || undefined }),
      spacing: { after: 20, line: 252 } })),
  });
  return new Table({
    width: { size: CONTENT_DXA, type: WidthType.DXA }, columnWidths: cols,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, i, false)) }))],
  });
}

async function figure(rel, caption, widthPx = MAX_IMG_PX) {
  const file = path.join(ROOT, rel);
  const meta = await sharp(file).metadata();
  const w = Math.min(widthPx, MAX_IMG_PX);
  const h = Math.round((meta.height / meta.width) * w);
  const type = rel.toLowerCase().endsWith(".jpg") || rel.toLowerCase().endsWith(".jpeg") ? "jpg" : "png";
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, children: [new ImageRun({
      type, data: fs.readFileSync(file), transformation: { width: w, height: h },
      altText: { title: caption, description: caption, name: path.basename(rel) } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: runs(caption, { italics: true, size: 18, color: INK2 }) }),
  ];
}

// borderless key/value table (tab stops are dropped by some renderers)
function metaTable(rows) {
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const borders = { top: none, bottom: none, left: none, right: none };
  const cols = [2300, CONTENT_DXA - 2300];
  const cell = (t, i) => new TableCell({ width: { size: cols[i], type: WidthType.DXA }, borders, margins: { top: 40, bottom: 40, left: 0, right: 80 },
    children: [new Paragraph({ style: "Normal", spacing: { after: 0 }, children: [new TextRun({ text: t, size: 20, font: "Arial", bold: i === 0, color: i === 0 ? INK2 : INK })] })] });
  return new Table({ width: { size: CONTENT_DXA, type: WidthType.DXA }, columnWidths: cols,
    borders: { top: none, bottom: none, left: none, right: none, insideHorizontal: none, insideVertical: none },
    rows: rows.map(([k, v]) => new TableRow({ children: [cell(k, 0), cell(v, 1)] })) });
}

function titlePage(o) {
  const line = (t, size, extra = {}) => new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 120 },
    children: [new TextRun({ text: t, size, font: "Arial", ...extra })] });
  return [
    new Paragraph({ spacing: { before: 1400 }, children: [] }),
    line("GENESIS", 28, { bold: true, color: ACCENT, characterSpacing: 40 }),
    line("Emergent social organisation in survival-only learning agents", 22, { color: INK2 }),
    new Paragraph({ spacing: { before: 600, after: 200 }, border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: ACCENT, space: 8 } },
      children: [new TextRun({ text: o.title, size: 48, bold: true, color: INK, font: "Arial" })] }),
    line(o.subtitle, 24, { color: INK2 }),
    new Paragraph({ spacing: { before: 900 }, children: [] }),
    metaTable([
      ["Course", o.course],
      ["Institute", "MPSTME, NMIMS (Vertically Integrated Projects vertical)"],
      ["Domain", "Artificial Intelligence & Data Systems"],
      ["Team", "Heet Ranpura (Sem V), Advik Divekar (Sem V)"],
      ["Faculty mentor", "[to fill]"],
      ["Repository", "github.com/HeetRanpura/Genesis"],
      ["Version / date", o.version || "1.0, 27 October 2026"],
    ]),
    pageBreak(),
  ];
}

function docOf(title, children) {
  return new Document({
    creator: "Heet Ranpura, Advik Divekar", title, description: `GENESIS VIP report: ${title}`,
    styles: {
      default: { document: { run: { font: "Arial", size: 21, color: INK } } },
      paragraphStyles: [
        { id: "Normal", name: "Normal", quickFormat: true, run: { font: "Arial", size: 21, color: INK, bold: false },
          paragraph: { spacing: { after: 120, line: 288 } } },
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 32, bold: true, font: "Arial", color: ACCENT }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0 } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 25, bold: true, font: "Arial", color: INK }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
        { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 22, bold: true, font: "Arial", color: INK2 }, paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 2 } },
      ],
    },
    numbering: { config: [
      { reference: "bullets", levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 270 } } } }] },
      { reference: "numbers", levels: [
        { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
    ] },
    sections: [{
      properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
      headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
        children: [new TextRun({ text: `GENESIS  |  ${title}`, size: 16, color: MUTED })] })] }) },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES], size: 16, color: MUTED })] })] }) },
      children,
    }],
  });
}

async function save(rel, title, children) {
  const file = path.join(OUT, rel);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, await Packer.toBuffer(docOf(title, children)));
  console.log("wrote", path.relative(ROOT, file));
}

module.exports = { P, H1, H2, H3, bullets, numbered, pageBreak, callout, table, figure, titlePage, save, runs, ROOT };
