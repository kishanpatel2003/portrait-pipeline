// Budget guard behavior tests — external contract only.
// Run: node --test pipeline/tests/
import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

process.env.BUDGET_FILE = join(mkdtempSync(join(tmpdir(), "budget-")), "allowance.json");
const guard = await import("../scripts/budget_guard.mjs");

beforeEach(() => guard.setAllowance(0));

test("default state blocks everything (hiatus)", () => {
  assert.equal(guard.remaining(), 0);
  assert.throws(() => guard.authorize(0.04, "one kontext image"), /allowance/i);
});

test("estimate above allowance is refused, at-or-below passes", () => {
  guard.setAllowance(1.0);
  assert.throws(() => guard.authorize(1.01, "too big"), /allowance/i);
  assert.doesNotThrow(() => guard.authorize(1.0, "exactly fits"));
});

test("debit decrements and is persisted across process-like reloads", async () => {
  guard.setAllowance(5);
  guard.debit(0.4, "X6a solo edit");
  guard.debit(0.1, "X6a extra");
  assert.equal(guard.remaining(), 4.5);
  // fresh import path simulates a new process reading the same file
  const fresh = await import("../scripts/budget_guard.mjs?reload=" + Date.now());
  assert.equal(fresh.remaining(), 4.5);
});

test("ledger records every debit with label", () => {
  guard.setAllowance(2);
  guard.debit(0.25, "X1d batch");
  const state = JSON.parse(readFileSync(process.env.BUDGET_FILE, "utf8"));
  assert.equal(state.ledger.at(-1).label, "X1d batch");
  assert.equal(state.ledger.at(-1).usd, 0.25);
});

test("authorize never spends — only debit moves the balance", () => {
  guard.setAllowance(1);
  guard.authorize(0.5, "estimate only");
  assert.equal(guard.remaining(), 1);
});
