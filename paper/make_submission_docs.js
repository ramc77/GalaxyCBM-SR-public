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
      P('Astronomy and Computing', { bold: true, after: 320 }),

      P('Dear Editor,', { after: 240 }),

      P('I submit for your consideration the manuscript entitled '
        + '"Interpretable Machine Learning for Galaxy Morphology: A Calibrated Symbolic '
        + 'Concept Bottleneck, and Why Its Explanations Do Not Transfer Across Surveys" '
        + 'for publication in Astronomy and Computing.'),

      P('Deep convolutional networks now classify galaxy morphology at survey scale, '
        + 'but they reason in representations with no physical referent, and the post-hoc '
        + 'attribution methods used to explain them describe a surrogate rather than the '
        + 'deployed decision. This manuscript takes the alternative route: it constrains '
        + 'the model to be interpretable by construction. I present the first Concept '
        + 'Bottleneck Model for galaxy morphology, in which a Zoobot ConvNeXt backbone '
        + 'predicts seventeen physically named concepts (ten Galaxy Zoo decision-tree '
        + 'tasks and seven statmorph structural statistics), and I replace the usual '
        + 'dense linear head with a compact analytic rule set discovered by symbolic '
        + 'regression. The deployed classifier is seven equations totalling 153 operator '
        + 'nodes, printed in full in the paper.'),

      P('The sample is built from twelve Galaxy Zoo Evo shards: 265,692 objects '
        + 'processed, 137,876 surviving the statmorph quality cuts, and 40,498 retained '
        + 'after Hubble-type derivation at a 0.5 clean-sample vote threshold. That '
        + 'threshold discards 70.6% of the objects reaching it and biases the surviving '
        + 'sample toward morphologically unambiguous systems, so absolute accuracies here '
        + 'are not directly comparable with magnitude-limited studies. All methods are '
        + 'therefore compared on identical splits rather than against literature values, '
        + 'and the manuscript states this explicitly.'),

      P('On 40,498 Galaxy Zoo Evo galaxies I measure what this interpretability costs '
        + 'rather than asserting it is free. An unconstrained ConvNeXt reaches a Cohen '
        + 'kappa of 0.601 against the symbolic head 0.571, so the concept bottleneck gives '
        + 'up 0.030. The symbolic head, a dense linear head and gradient-boosted trees over '
        + 'the same concepts agree to within 0.002, which localises that cost to the '
        + 'bottleneck rather than to the symbolic form of the classifier. A practitioner who '
        + 'accepts the concept layer for auditability can therefore adopt the analytic rule '
        + 'set at no further accuracy penalty, in place of a 15-million-parameter network. '
        + 'On macro-F1, which weights the rare morphologies equally, the ordering reverses '
        + 'and every concept-based model exceeds the network.'),

      P('Two further results are, in my view, of wider consequence than the accuracy '
        + 'comparison, and they are the reason I believe this work suits your readership. '
        + 'First, split-conformal calibration is marginally exact (88.9% empirical against '
        + 'a 90% target) while per-class coverage ranges from 0.98 for ellipticals to 0.08 '
        + 'for the rarest spiral class. The guarantee is satisfied almost entirely by '
        + 'over-covering the dominant class, which directly threatens rare-object science; '
        + 'reporting marginal coverage alone is therefore misleading on the imbalanced '
        + 'samples typical of morphology catalogues. Second, applying the frozen pipeline '
        + 'to 161,395 Euclid Q1 galaxies preserves classification accuracy while a symbolic '
        + 'refit on Euclid recovers under half of the reference rules’ concept usage. '
        + 'Accuracy transfer and interpretation transfer are separate properties, and an '
        + 'accuracy-only robustness audit would certify a pipeline whose explanations had '
        + 'silently changed.'),

      P('I believe this work fits the scope of Astronomy and Computing directly. Its '
        + 'contribution is methodological rather than a new astrophysical measurement: a '
        + 'released and reproducible classification pipeline, together with two diagnostics '
        + 'that other groups can apply to their own morphology models. The rule-stability '
        + 'index and the per-class conformal coverage audit are both offered as reusable '
        + 'evaluation tools, and I hope they are of use to the community independently of '
        + 'the particular architecture reported here.'),

      P('All code, configurations, random seeds and derived results are released publicly. '
        + 'Every number quoted in the manuscript is written to a machine-readable results '
        + 'file by the evaluation stage and checked against a manifest of paper claims at '
        + 'build time, so the reported values can be regenerated from a single command. The '
        + 'manuscript also states its limitations plainly, including the severe class '
        + 'imbalance that bounds conclusions about the two rarest morphologies and the '
        + 'uncontrolled nature of the cross-survey comparison.'),

      P('This manuscript is original, has not been published previously, and is not under '
        + 'consideration by any other journal. I am the sole author and declare no competing '
        + 'interests. [ADD IF APPLICABLE: suggested reviewers and any opposed reviewers.]'),

      P('Thank you for considering this work.', { after: 320 }),

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
