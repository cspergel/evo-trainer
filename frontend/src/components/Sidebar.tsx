import { colors } from "../lib/tokens";

interface NavItem { label: string; active?: boolean }
interface NavSection { label: string; items: NavItem[] }

const navSections: NavSection[] = [
  {
    label: "Overview",
    items: [{ label: "Dashboard", active: true }],
  },
  {
    label: "Trading",
    items: [
      { label: "Trade History" },
      { label: "Active Positions" },
      { label: "Strategy Performance" },
    ],
  },
  {
    label: "Signals",
    items: [
      { label: "Signal Feed" },
      { label: "Source Health" },
      { label: "Source Lifecycle" },
    ],
  },
  {
    label: "Evolution",
    items: [
      { label: "Evolution Events" },
      { label: "Skill Lineage" },
      { label: "Fitness History" },
    ],
  },
  {
    label: "System",
    items: [
      { label: "LLM Costs" },
      { label: "Risk Constraints" },
      { label: "Settings" },
    ],
  },
];

export function Sidebar() {
  return (
    <aside
      style={{
        width: 240,
        minHeight: "100vh",
        background: colors.surface,
        borderRight: `1px solid ${colors.border}`,
        padding: "16px 0",
        position: "fixed",
        left: 0,
        top: 0,
        overflowY: "auto",
      }}
    >
      {/* Logo */}
      <div style={{ padding: "0 20px 20px", display: "flex", alignItems: "center", gap: 8 }}>
        <div
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: colors.accent,
          }}
        />
        <span style={{ fontWeight: 700, fontSize: 15, color: colors.text }}>
          Evolve-Trader
        </span>
      </div>

      {/* Nav sections */}
      {navSections.map((section) => (
        <div key={section.label} style={{ marginBottom: 8 }}>
          <div
            style={{
              padding: "6px 20px",
              fontSize: 11,
              fontWeight: 600,
              color: colors.textMuted,
              textTransform: "uppercase",
              letterSpacing: "0.05em",
            }}
          >
            {section.label}
          </div>
          {section.items.map((item) => (
            <div
              key={item.label}
              style={{
                padding: "7px 20px",
                fontSize: 13,
                color: item.active ? colors.accentText : colors.textSecondary,
                background: item.active ? colors.accentSoft : "transparent",
                borderLeft: item.active ? `3px solid ${colors.accent}` : "3px solid transparent",
                fontWeight: item.active ? 600 : 400,
                cursor: "pointer",
              }}
            >
              {item.label}
            </div>
          ))}
        </div>
      ))}
    </aside>
  );
}
