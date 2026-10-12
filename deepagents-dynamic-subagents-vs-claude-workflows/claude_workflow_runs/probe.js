export const meta = {
  name: 'ticket-triage-probe',
  description: 'Research probe: classify 4 support tickets then route each to a specialist (compare with deepagents dynamic subagents)',
  phases: [
    { title: 'Probes', detail: 'failure semantics, no agents' },
    { title: 'Classify', detail: 'one classifier per ticket' },
    { title: 'Handle', detail: 'route to specialist by label' },
  ],
}

phase('Probes')
const probes = {}
probes.parallelWithThrowingThunk = await parallel([
  async () => 'ok-1',
  async () => { throw new Error('boom') },
  async () => 'ok-3',
])
log('probes: ' + JSON.stringify(probes))

const SPECIALIST = {
  bug: 'You are a bug triage specialist. In one sentence give the likely cause and the next step.',
  feature: 'You are a product analyst. In one sentence rate effort (S/M/L) and impact.',
  question: 'You are a support specialist. In one sentence answer and say where in the product to look.',
}
const CLASS = { type: 'object', properties: { category: { type: 'string', enum: ['bug', 'feature', 'question'] } }, required: ['category'] }
const OUTCOME = { type: 'object', properties: { outcome: { type: 'string' } }, required: ['outcome'] }

const results = await pipeline(
  args.tickets,
  (t) => agent(`Classify this support ticket as exactly one of bug, feature, question. Do not use any tools.\nTicket ${t.id}: ${t.text}`,
    { label: `classify ${t.id}`, phase: 'Classify', schema: CLASS, model: 'haiku', effort: 'low' }),
  (cls, t) => agent(`${SPECIALIST[cls.category]} Do not use any tools; answer from general knowledge.\nTicket ${t.id}: ${t.text}`,
    { label: `${t.id} -> ${cls.category}`, phase: 'Handle', schema: OUTCOME, model: 'haiku', effort: 'low' })
    .then((o) => ({ id: t.id, category: cls.category, outcome: o.outcome })),
)
return { probes, results }