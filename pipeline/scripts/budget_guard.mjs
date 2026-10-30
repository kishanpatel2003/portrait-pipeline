// Budget guard — every paid fal call must pass through here.
// Allowance persists in pipeline/budget_allowance.json and DEFAULTS TO $0
// (spending hiatus is the resting state, not a flag). Raise it explicitly:
//   node pipeline/scripts/budget_guard.mjs set 3.50
//   node pipeline/scripts/budget_guard.mjs status
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const FILE = process.env.BUDGET_FILE ?? resolve(join(HERE, "..", "budget_allowance.json"));

const load = () => {
  if (!existsSync(FILE)) return { allowance_usd: 0, ledger: [] };
  return JSON.parse(readFileSync(FILE, "utf8"));
};
const save = (state) => {
  mkdirSync(dirname(FILE), { recursive: true });
  writeFileSync(FILE, JSON.stringify(state, null, 1));
};

export function remaining() {
  const s = load();
  return Math.round((s.allowance_usd - s.ledger.reduce((a, e) => a + e.usd, 0)) * 100) / 100;
}

export function setAllowance(usd) {
  save({ allowance_usd: Number(usd), ledger: [] });
}

// Pre-flight check: refuses when the estimate exceeds what's left. Spends nothing.
export function authorize(estimateUsd, label) {
  const left = remaining();
  if (estimateUsd > left) {
    throw new Error(
      `budget guard: "${label}" estimated at $${estimateUsd.toFixed(2)} but allowance has ` +
      `$${left.toFixed(2)} left. Raise it first: node pipeline/scripts/budget_guard.mjs set <usd>`);
  }
  return left;
}

export function debit(usd, label) {
  const s = load();
  s.ledger.push({ usd: Number(usd), label, ts: new Date().toISOString() });
  save(s);
  return remaining();
}

// CLI
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [cmd, arg] = process.argv.slice(2);
  if (cmd === "set") { setAllowance(arg); console.log(`allowance set to $${Number(arg).toFixed(2)}`); }
  else if (cmd === "log") console.log(load().ledger.map(e => `${e.ts}  $${e.usd.toFixed(2)}  ${e.label}`).join("\n") || "(no spend recorded)");
  else console.log(`remaining allowance: $${remaining().toFixed(2)} (file: ${FILE})`);
}
