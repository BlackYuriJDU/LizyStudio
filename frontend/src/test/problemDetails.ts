import { ApiError } from "@/api/client";
import type { ProblemDetailsBody } from "@/api/errors";
import type { ConfigError } from "@/api/types";

/**
 * Test builders for the RFC 9457 Problem Details bodies the workspace
 * endpoints answer with (P-0110). The shapes are copied from real backend
 * output (`PUT /api/workspace/config` with `split.n_splits=1000` on a
 * 50-row dataset, and with a running job holding the slot), not written
 * from the type, so a test that passes here passes against the server.
 */

/** The `validation_errors` entry the real backend sent for n_splits=1000. */
export const N_SPLITS_ERROR: ConfigError = {
  path: "split.n_splits",
  message:
    "n_splits=1000 is greater than the number of samples in the loaded dataset (n_rows=50). Reduce Folds to at most 50.",
  severity: "error",
  suggested_fix: "Set Folds (split.n_splits) to 50 or fewer.",
};

/** 422 `VALIDATION_ERROR` body: the config was not saved. */
export function validationProblemBody(
  validationErrors: ConfigError[] = [N_SPLITS_ERROR],
): ProblemDetailsBody {
  return {
    type: "about:blank",
    title: "Unprocessable Entity",
    status: 422,
    detail: "Config validation failed",
    code: "VALIDATION_ERROR",
    details: {},
    validation_errors: validationErrors as unknown as {
      [key: string]: unknown;
    }[],
  };
}

/** The `ApiError` the API client throws for a 422 validation rejection. */
export function validationProblem(
  validationErrors: ConfigError[] = [N_SPLITS_ERROR],
): ApiError {
  return new ApiError(422, validationProblemBody(validationErrors));
}

/** The `ApiError` for a 409 `WORKSPACE_LOCKED` refusal of a config write. */
export function workspaceLockedProblem(jobId = "job_running"): ApiError {
  return new ApiError(409, {
    type: "about:blank",
    title: "Conflict",
    status: 409,
    detail: `Config is locked while job ${jobId} is running`,
    code: "WORKSPACE_LOCKED",
    details: { job_id: jobId },
    validation_errors: null,
  } satisfies ProblemDetailsBody);
}
