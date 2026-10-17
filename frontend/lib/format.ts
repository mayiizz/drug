// Lightweight formatters for CURA candidate display. Single source of truth
// for unit-correct binding labels so chips and exports stay in sync.

export function formatKd(nm: number | null | undefined): string {
  if (nm == null || Number.isNaN(nm)) return "—";
  if (nm < 1) return "< 1 nM";
  if (nm >= 1000) return `${(nm / 1000).toFixed(1)} \u00b5M`;
  return `${nm.toFixed(0)} nM`;
}

export function formatKdLabel(nm: number | null | undefined): string {
  if (nm == null || Number.isNaN(nm)) return "Kd \u2248 —";
  return `Kd \u2248 ${formatKd(nm)}`;
}
