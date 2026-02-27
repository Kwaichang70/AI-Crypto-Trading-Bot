import type { RunStatus, OrderStatus } from "@/lib/types";

type BadgeVariant = "success" | "warning" | "danger" | "neutral" | "info";

interface StatusBadgeProps {
  status: string;
  variant?: BadgeVariant;
}

const RUN_STATUS_VARIANTS: Record<RunStatus, BadgeVariant> = {
  running: "success",
  stopped: "neutral",
  error: "danger",
};

const ORDER_STATUS_VARIANTS: Record<OrderStatus, BadgeVariant> = {
  new: "neutral",
  pending_submit: "info",
  open: "info",
  partial: "warning",
  filled: "success",
  canceled: "neutral",
  rejected: "danger",
  expired: "neutral",
};

const VARIANT_CLASSES: Record<BadgeVariant, string> = {
  success: "badge-success",
  warning: "badge-warning",
  danger: "badge-danger",
  neutral: "badge-neutral",
  info: "inline-flex items-center rounded-full bg-blue-100 px-2.5 py-0.5 text-xs font-medium text-blue-800 dark:bg-blue-900/30 dark:text-blue-400",
};

function inferVariant(status: string): BadgeVariant {
  const runVariant = RUN_STATUS_VARIANTS[status as RunStatus];
  if (runVariant) return runVariant;
  const orderVariant = ORDER_STATUS_VARIANTS[status as OrderStatus];
  if (orderVariant) return orderVariant;
  return "neutral";
}

export function StatusBadge({ status, variant }: StatusBadgeProps) {
  const resolvedVariant = variant ?? inferVariant(status);
  const classes = VARIANT_CLASSES[resolvedVariant];

  return (
    <span className={classes}>
      {resolvedVariant === "success" && (
        <span className="mr-1.5 inline-block h-1.5 w-1.5 rounded-full bg-green-500" aria-hidden="true" />
      )}
      {resolvedVariant === "danger" && (
        <span className="mr-1.5 inline-block h-1.5 w-1.5 rounded-full bg-red-500" aria-hidden="true" />
      )}
      {status}
    </span>
  );
}

/** Infer badge variant from run status string. */
export function RunStatusBadge({ status }: { status: string }) {
  return <StatusBadge status={status} variant={inferVariant(status)} />;
}
