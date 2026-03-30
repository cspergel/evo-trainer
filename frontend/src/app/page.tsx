"use client";

import { useEffect, useState } from "react";

interface SystemStatus {
  mode: string;
  total_trades: number;
  total_signals: number;
  total_evolution_events: number;
  kill_switch_status: string;
  approval_queue_size: number;
  evolution_paused: boolean;
}

interface Portfolio {
  total_value: number;
  cash: number;
  positions: Record<string, unknown>;
  drawdown: number;
  sharpe_ratio: number | null;
}

interface Trade {
  id: number;
  strategy_skill: string;
  ticker: string;
  direction: string;
  pnl: number | null;
  return_pct: number | null;
  entry_date: string | null;
}

interface Signal {
  id: number;
  source: string;
  source_entity: string;
  confidence: number;
  signal_type: string;
  timestamp: string;
}

interface EvolutionEvent {
  id: number;
  event_type: string;
  parent_skill: string | null;
  child_skill: string;
  trigger_reason: string;
  created_at: string;
}

interface Costs {
  total_cost_usd: number;
  cost_by_component: Record<string, number>;
}

const API = "http://localhost:8100";

function useFetch<T>(url: string, fallback: T): T {
  const [data, setData] = useState<T>(fallback);
  useEffect(() => {
    fetch(url)
      .then((r) => r.json())
      .then(setData)
      .catch(() => {});
  }, [url]);
  return data;
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-3">
        {title}
      </h2>
      {children}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <div className="text-2xl font-bold text-white">{value}</div>
      <div className="text-xs text-gray-500">{label}</div>
    </div>
  );
}

export default function Dashboard() {
  const status = useFetch<SystemStatus>(`${API}/api/status`, {
    mode: "loading...",
    total_trades: 0,
    total_signals: 0,
    total_evolution_events: 0,
    kill_switch_status: "unknown",
    approval_queue_size: 0,
    evolution_paused: false,
  });

  const portfolio = useFetch<Portfolio>(`${API}/api/portfolio/latest`, {
    total_value: 0,
    cash: 0,
    positions: {},
    drawdown: 0,
    sharpe_ratio: null,
  });

  const trades = useFetch<Trade[]>(`${API}/api/trades?limit=10`, []);
  const signals = useFetch<Signal[]>(`${API}/api/signals?limit=10`, []);
  const evolution = useFetch<EvolutionEvent[]>(`${API}/api/evolution?limit=10`, []);
  const costs = useFetch<Costs>(`${API}/api/costs`, {
    total_cost_usd: 0,
    cost_by_component: {},
  });

  return (
    <div className="space-y-6">
      {/* System Status Bar */}
      <div className="flex gap-4 items-center text-sm">
        <span className="px-2 py-1 bg-blue-900/50 text-blue-300 rounded text-xs font-mono">
          {status.mode}
        </span>
        <span className="text-gray-500">
          {status.total_trades} trades · {status.total_signals} signals · {status.total_evolution_events} evolution events
        </span>
      </div>

      {/* Top row: Portfolio + Costs */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card title="Portfolio Health">
          <div className="grid grid-cols-2 gap-4">
            <Stat label="Total Value" value={`$${portfolio.total_value.toLocaleString()}`} />
            <Stat label="Cash" value={`$${portfolio.cash.toLocaleString()}`} />
            <Stat label="Drawdown" value={`${(portfolio.drawdown * 100).toFixed(1)}%`} />
            <Stat label="Sharpe" value={portfolio.sharpe_ratio?.toFixed(2) ?? "—"} />
          </div>
        </Card>

        <Card title="LLM Costs">
          <Stat label="Total Spend" value={`$${costs.total_cost_usd.toFixed(3)}`} />
          <div className="mt-3 space-y-1">
            {Object.entries(costs.cost_by_component).map(([comp, cost]) => (
              <div key={comp} className="flex justify-between text-xs">
                <span className="text-gray-400">{comp}</span>
                <span className="text-gray-300">${cost.toFixed(4)}</span>
              </div>
            ))}
          </div>
        </Card>

        <Card title="Operator Status">
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">Kill Switch</span>
              <span className="text-yellow-500">{status.kill_switch_status}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Approval Queue</span>
              <span className="text-gray-300">{status.approval_queue_size}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Evolution</span>
              <span className={status.evolution_paused ? "text-red-400" : "text-green-400"}>
                {status.evolution_paused ? "Paused" : "Active"}
              </span>
            </div>
          </div>
        </Card>
      </div>

      {/* Middle row: Trades + Signals */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card title="Recent Trades">
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-gray-500 border-b border-gray-800">
                  <th className="text-left py-1">Ticker</th>
                  <th className="text-left py-1">Strategy</th>
                  <th className="text-left py-1">Dir</th>
                  <th className="text-right py-1">PnL</th>
                </tr>
              </thead>
              <tbody>
                {trades.map((t) => (
                  <tr key={t.id} className="border-b border-gray-800/50">
                    <td className="py-1 font-mono">{t.ticker}</td>
                    <td className="py-1 text-gray-400">{t.strategy_skill}</td>
                    <td className={`py-1 ${t.direction === "BUY" ? "text-green-400" : "text-red-400"}`}>
                      {t.direction}
                    </td>
                    <td className={`py-1 text-right ${(t.pnl ?? 0) >= 0 ? "text-green-400" : "text-red-400"}`}>
                      {t.pnl != null ? `$${t.pnl.toFixed(0)}` : "—"}
                    </td>
                  </tr>
                ))}
                {trades.length === 0 && (
                  <tr><td colSpan={4} className="py-2 text-gray-600 text-center">No trades yet</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="Recent Signals">
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-gray-500 border-b border-gray-800">
                  <th className="text-left py-1">Source</th>
                  <th className="text-left py-1">Entity</th>
                  <th className="text-left py-1">Type</th>
                  <th className="text-right py-1">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {signals.map((s) => (
                  <tr key={s.id} className="border-b border-gray-800/50">
                    <td className="py-1 font-mono">{s.source}</td>
                    <td className="py-1 text-gray-400">{s.source_entity}</td>
                    <td className="py-1">{s.signal_type}</td>
                    <td className="py-1 text-right text-blue-400">{(s.confidence * 100).toFixed(0)}%</td>
                  </tr>
                ))}
                {signals.length === 0 && (
                  <tr><td colSpan={4} className="py-2 text-gray-600 text-center">No signals yet</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      {/* Bottom row: Evolution */}
      <Card title="Evolution Events">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-gray-500 border-b border-gray-800">
                <th className="text-left py-1">Type</th>
                <th className="text-left py-1">Parent</th>
                <th className="text-left py-1">Child</th>
                <th className="text-left py-1">Reason</th>
              </tr>
            </thead>
            <tbody>
              {evolution.map((e) => (
                <tr key={e.id} className="border-b border-gray-800/50">
                  <td className="py-1">
                    <span className={`px-1.5 py-0.5 rounded text-xs font-mono ${
                      e.event_type === "FIX" ? "bg-yellow-900/50 text-yellow-300" :
                      e.event_type === "DERIVED" ? "bg-blue-900/50 text-blue-300" :
                      "bg-green-900/50 text-green-300"
                    }`}>
                      {e.event_type}
                    </span>
                  </td>
                  <td className="py-1 text-gray-400 font-mono">{e.parent_skill ?? "—"}</td>
                  <td className="py-1 font-mono">{e.child_skill}</td>
                  <td className="py-1 text-gray-400 truncate max-w-xs">{e.trigger_reason}</td>
                </tr>
              ))}
              {evolution.length === 0 && (
                <tr><td colSpan={4} className="py-2 text-gray-600 text-center">No evolution events yet</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
