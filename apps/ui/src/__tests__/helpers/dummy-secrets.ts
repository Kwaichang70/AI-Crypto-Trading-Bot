import { randomBytes } from "crypto";

/**
 * PR-HYG-01: a dummy secret or key in a test is built at run time, never
 * committed as a literal. Compare against the same constant; never inline it.
 */
export const dummySecret = (label: string): string => `${label}-${randomBytes(16).toString("hex")}`;
