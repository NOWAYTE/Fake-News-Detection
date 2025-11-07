"use client";

import { useState } from "react";

type Analysis = {
  text: string;
  prediction: "real" | "fake" | string;
  confidence: number; // confidence of the predicted class
  author?: string;
  created_at?: string;
};

export default function HomePage() {
  const [url, setUrl] = useState("");
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isTwitterUrl = (u: string) => {
    try {
      const parsed = new URL(u);
      return /(^|\.)twitter\.com$|(^|\.)x\.com$/.test(parsed.hostname);
    } catch {
      return false;
    }
  };

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setAnalysis(null);

    if (!isTwitterUrl(url)) {
      setError("Please paste a valid Twitter/X URL.");
      return;
    }

    setLoading(true);
    try {
      const base = process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, "") || "";
      const endpoint = `${base}/analyze`;
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      if (!res.ok) throw new Error("Failed to analyze. Try again.");
      const data: Analysis = await res.json();
      setAnalysis(data);
    } catch (err: any) {
      setError(err?.message ?? "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  // Convert prediction+confidence into real/fake percentages
  const realPct = analysis
    ? Math.round(((analysis.prediction === "real" ? analysis.confidence : 1 - analysis.confidence) || 0) * 100)
    : 0;
  const fakePct = analysis ? 100 - realPct : 0;

  return (
    <main className="min-h-screen bg-background text-foreground">
      <section className="mx-auto flex max-w-3xl flex-col gap-6 px-4 py-12">
        <div className="space-y-2">
          <h1 className="text-3xl font-semibold tracking-tight">Fake News Detector</h1>
          <p className="text-sm text-muted-foreground">
            Paste a Twitter/X post URL. We'll fetch the content and estimate whether it's real or fake.
          </p>
        </div>

        <form onSubmit={onSubmit} className="flex w-full items-center gap-2">
          <input
            type="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://twitter.com/... or https://x.com/..."
            className="flex-1 rounded-md border border-border bg-background px-3 py-2 text-sm outline-none ring-offset-background focus:ring-2 focus:ring-ring"
          />
          <button
            type="submit"
            disabled={loading}
            className="inline-flex items-center justify-center whitespace-nowrap rounded-md bg-foreground px-4 py-2 text-sm font-medium text-background shadow transition-colors hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "Analyzing..." : "Analyze"}
          </button>
        </form>

        {error && (
          <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            {error}
          </div>
        )}

        {analysis && (
          <div className="grid gap-4 md:grid-cols-5">
            <div className="md:col-span-3 space-y-3 rounded-lg border bg-card p-4">
              <div className="text-xs text-muted-foreground">Tweet content</div>
              <div className="whitespace-pre-wrap text-sm leading-6">{analysis.text}</div>
              <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
                {analysis.author && <span>Author: {analysis.author}</span>}
                {analysis.created_at && <span>Posted: {new Date(analysis.created_at).toLocaleString()}</span>}
              </div>
            </div>
            <div className="md:col-span-2 space-y-3 rounded-lg border bg-card p-4">
              <div className="text-xs text-muted-foreground">Assessment</div>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium">Real</span>
                  <span className="tabular-nums">{realPct}%</span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded bg-muted">
                  <div className="h-full bg-green-600" style={{ width: `${realPct}%` }} />
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium">Fake</span>
                  <span className="tabular-nums">{fakePct}%</span>
                </div>
                <div className="h-2 w-full overflow-hidden rounded bg-muted">
                  <div className="h-full bg-red-600" style={{ width: `${fakePct}%` }} />
                </div>
              </div>
            </div>
          </div>
        )}
      </section>
    </main>
  );
}

