import { colors, fonts } from "../lib/tokens";

interface Column {
  key: string;
  label: string;
  align?: "left" | "right";
  mono?: boolean;
  render?: (value: unknown, row: Record<string, unknown>) => React.ReactNode;
}

interface DataTableProps {
  columns: Column[];
  rows: Record<string, unknown>[];
  emptyMessage?: string;
}

export function DataTable({ columns, rows, emptyMessage = "No data" }: DataTableProps) {
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            {columns.map((col) => (
              <th
                key={col.key}
                style={{
                  textAlign: col.align || "left",
                  fontSize: 11,
                  fontWeight: 600,
                  color: colors.textMuted,
                  textTransform: "uppercase",
                  letterSpacing: "0.03em",
                  padding: "6px 8px",
                  borderBottom: `1px solid ${colors.border}`,
                }}
              >
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td
                colSpan={columns.length}
                style={{
                  textAlign: "center",
                  padding: 24,
                  color: colors.textMuted,
                  fontSize: 13,
                }}
              >
                {emptyMessage}
              </td>
            </tr>
          ) : (
            rows.map((row, i) => (
              <tr key={i}>
                {columns.map((col) => (
                  <td
                    key={col.key}
                    style={{
                      textAlign: col.align || "left",
                      fontSize: 13,
                      fontFamily: col.mono ? fonts.mono : fonts.body,
                      color: colors.text,
                      padding: "8px",
                      borderBottom: `1px solid ${colors.borderSoft}`,
                    }}
                  >
                    {col.render
                      ? col.render(row[col.key], row)
                      : String(row[col.key] ?? "—")}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}
