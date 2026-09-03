const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  LevelFormat, convertInchesToTwip,
} = require('docx');
const fs = require('fs');

const P = (text, opts = {}) => new Paragraph({
  spacing: { after: opts.after ?? 160, line: opts.line ?? 276 },
  alignment: opts.align,
  children: [new TextRun({ text, bold: opts.bold, italics: opts.italics, size: opts.size ?? 22 })],
});

const BLANK = () => new Paragraph({ children: [new TextRun('')], spacing: { after: 120 } });

// ---------------------------------------------------------------- highlights
// Elsevier: 3-5 bullets, each <= 85 characters including spaces.
const highlightBullets = [
  'First concept bottleneck model for galaxy morphology classification',
  'The concept bottleneck costs 0.030 in kappa; the symbolic head adds no further cost',
  'A 153-node analytic rule set replaces a 15M-parameter network and is printed in full',
  'Per-class coverage spans 0.98 to 0.08; the fix triples prediction-set size',
  'Accuracy transfers to Euclid Q1 but the symbolic explanations do not (J = 0.42)',
];
highlightBullets.forEach((b) => {
  if (b.length > 85) throw new Error(`Highlight exceeds 85 chars (${b.length}): ${b}`);
});

const highlights = new Document({
  numbering: {
    config: [{
      reference: 'hl-bullets',
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: '•',
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: convertInchesToTwip(0.3), hanging: convertInchesToTwip(0.2) } } },
      }],
    }],
  },
  sections: [{
    properties: { page: { margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    children: [
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        spacing: { after: 200 },
        children: [new TextRun({ text: 'Highlights', bold: true, size: 28 })],
      }),
      P('Interpretable Machine Learning for Galaxy Morphology: A Calibrated '
        + 'Symbolic Concept Bottleneck, and Why Its Explanations Do Not Transfer '
        + 'Across Surveys', { bold: true }),
      P('Ram Chand', { italics: true, after: 320 }),
      ...highlightBullets.map((t) => new Paragraph({
        numbering: { reference: 'hl-bullets', level: 0 },
        spacing: { after: 140, line: 276 },
        children: [new TextRun({ text: t, size: 22 })],
      })),
    ],
  }],
});

// -------------------------------------------------------------- cover letter
// Revision cover letter for Physics of the Dark Universe. This accompanies a
// resubmission after review, not a first submission, so it points at the
// response-to-reviewers document rather than re-pitching the paper cold.
const coverLetter = new Document({
  sections: [{
    properties: { page: { margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    children: [
      P('Ram Chand', { bold: true, after: 0 }),
      P('Department of Natural Sciences', { after: 0 }),
      P('The Begum Nusrat Bhutto Women University', { after: 0 }),
      P('Sukkur, Sindh, Pakistan', { after: 0 }),
      P('ram.chand2k11@yahoo.com', { after: 320 }),

      P('[DATE]', { after: 320 }),

      P('The Editor-in-Chief', { after: 0 }),
      P('Physics of the Dark Universe', { bold: true, after: 320 }),

      P('Dear Editor,', { after: 240 }),

      P('Please find enclosed a revised version of manuscript [MANUSCRIPT ID], '
        + '"Interpretable Machine Learning for Galaxy Morphology: A Calibrated Symbolic '
        + 'Concept Bottleneck, and Why Its Explanations Do Not Transfer Across Surveys," '
        + 'submitted for reconsideration following the reviewer’s report. I thank the '
        + 'reviewer for a careful reading, and I have revised the manuscript to address '
        + 'every point raised. A full point-by-point response accompanies this letter as a '
        + 'separate document; I summarise the substance of the revision below.'),

      P('The central criticism was that a paper whose subject is interpretability must '
        + 'itself be interpretable by its readership, and that this manuscript, in its '
        + 'submitted form, was not: it described the classifier’s equations rather than '
        + 'printing them, introduced its concept vocabulary after using it, and moved '
        + 'through conformal prediction and cross-survey rule stability too quickly for a '
        + 'reader without a machine-learning background. I accept this criticism without '
        + 'reservation. The revision adds a new appendix printing all seven governing '
        + 'equations in full, with a symbol dictionary; states the complete seventeen-concept '
        + 'vocabulary at first use; and rewrites the sections on conformal prediction and '
        + 'survey shift in plain language, defining every term before it is used.'),

      P('Two results changed as a direct consequence of taking the review seriously rather '
        + 'than defending the original text. First, the reviewer asked for direct evidence '
        + 'that the discovered rules reproduce the Galaxy Zoo decision-tree structure; '
        + 'testing this mechanically showed the submitted claim was too strong. Six of seven '
        + 'rules factor through the root task rather than all seven, and the manuscript now '
        + 'reports the corrected, narrower finding. Second, the reviewer noted that the '
        + 'rare-class conformal coverage failure was identified but not addressed; I have '
        + 'now implemented class-conditional calibration rather than deferring it, and the '
        + 'result is a genuine trade rather than a clean fix—worst-case coverage rises from '
        + '0.083 to 0.885, but mean prediction-set size roughly triples and no test galaxy '
        + 'receives a unique label. Both findings are reported plainly, including where they '
        + 'revise or complicate the paper’s original claims.'),

      P('A third point, on whether the adopted symbolic rules are justified in size or are '
        + 'avoidable complexity, is answered with a new ablation: a rule set one-fifth the '
        + 'size matches the adopted one on held-out accuracy and Cohen’s kappa, and the '
        + 'manuscript now states this directly rather than asserting that the larger '
        + 'expressions are necessary.'),

      P('All revised numbers are traced to a machine-readable results file and checked '
        + 'against a manifest of paper claims at build time, and the two new analyses are '
        + 'released as part of the public code alongside the rest of the pipeline. I believe '
        + 'the manuscript is substantially strengthened by this process and hope it is now '
        + 'suitable for publication in Physics of the Dark Universe.'),

      P('This manuscript remains original, has not been published elsewhere, and is not '
        + 'under consideration by any other journal. I am the sole author and declare no '
        + 'competing interests.'),

      P('Thank you for considering this revision.', { after: 320 }),

      P('Yours sincerely,', { after: 480 }),
      P('Ram Chand', { bold: true, after: 0 }),
      P('Department of Natural Sciences', { after: 0 }),
      P('The Begum Nusrat Bhutto Women University, Sukkur, Sindh, Pakistan'),
    ],
  }],
});

(async () => {
  fs.writeFileSync('cover_letter.docx', await Packer.toBuffer(coverLetter));
  fs.writeFileSync('highlights.docx', await Packer.toBuffer(highlights));
  console.log('wrote cover_letter.docx and highlights.docx');
  highlightBullets.forEach((b) => console.log(`  [${String(b.length).padStart(2)} chars] ${b}`));
})();
