/**
 * apps/ui/src/__tests__/components/live-confirm-dialog.test.tsx
 * -----------------------------------------------------------------
 * WP1.7 synthesis-spec I11 / SY-10 (S13): the live-trading confirmation
 * token must never be persisted anywhere beyond the dialog's own lifetime,
 * and must be handed to the caller exactly once (for the caller to place it
 * on the `X-Live-Confirm-Token` request header — never a request body).
 */

import React, { useState } from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { LiveConfirmDialog } from "@/components/live-confirm-dialog";

function Harness({ onConfirm }: { onConfirm: (token: string) => void }) {
  const [open, setOpen] = useState(true);
  return (
    <LiveConfirmDialog
      open={open}
      title="Confirm"
      description="desc"
      loading={false}
      onCancel={() => setOpen(false)}
      onConfirm={(token) => {
        onConfirm(token);
        setOpen(false);
      }}
    />
  );
}

describe("LiveConfirmDialog", () => {
  it("hands the typed token to onConfirm exactly once, for the caller to send as a header", async () => {
    const onConfirm = jest.fn();
    render(<Harness onConfirm={onConfirm} />);

    const input = screen.getByLabelText(/Live trading confirmation token/i);
    fireEvent.change(input, { target: { value: "s3cr3t-token" } });
    fireEvent.click(screen.getByRole("button", { name: "Confirm" }));

    expect(onConfirm).toHaveBeenCalledTimes(1);
    expect(onConfirm).toHaveBeenCalledWith("s3cr3t-token");
  });

  it("clears the token field when the dialog is cancelled and reopened", async () => {
    function ReopenHarness() {
      const [open, setOpen] = useState(true);
      return (
        <>
          <button onClick={() => setOpen(true)}>reopen</button>
          <LiveConfirmDialog
            open={open}
            title="Confirm"
            description="desc"
            loading={false}
            onCancel={() => setOpen(false)}
            onConfirm={() => setOpen(false)}
          />
        </>
      );
    }
    render(<ReopenHarness />);

    const input = () => screen.getByLabelText(/Live trading confirmation token/i) as HTMLInputElement;
    fireEvent.change(input(), { target: { value: "leftover-token" } });
    expect(input().value).toBe("leftover-token");

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(screen.queryByLabelText(/Live trading confirmation token/i)).not.toBeInTheDocument());

    fireEvent.click(screen.getByText("reopen"));
    expect(input().value).toBe("");
  });

  it("never writes the token to localStorage or sessionStorage", () => {
    const onConfirm = jest.fn();
    render(<Harness onConfirm={onConfirm} />);
    fireEvent.change(screen.getByLabelText(/Live trading confirmation token/i), {
      target: { value: "s3cr3t-token" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Confirm" }));

    expect(JSON.stringify(window.localStorage)).not.toContain("s3cr3t-token");
    expect(JSON.stringify(window.sessionStorage)).not.toContain("s3cr3t-token");
    for (let i = 0; i < window.localStorage.length; i += 1) {
      const key = window.localStorage.key(i);
      expect(key ? window.localStorage.getItem(key) : null).not.toContain("s3cr3t-token");
    }
  });

  it("renders a password-type input so the token is not shoulder-surfable", () => {
    render(<Harness onConfirm={jest.fn()} />);
    const input = screen.getByLabelText(/Live trading confirmation token/i);
    expect(input).toHaveAttribute("type", "password");
  });
});
