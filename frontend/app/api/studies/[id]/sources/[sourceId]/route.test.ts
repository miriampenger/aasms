import { describe, it, expect, vi, beforeEach } from "vitest";
import { NextRequest } from "next/server";

const mockPrisma = vi.hoisted(() => ({
  source: { findFirst: vi.fn(), update: vi.fn() },
}));

vi.mock("@/lib/db", () => ({
  prisma: mockPrisma,
}));

import { PATCH } from "./route";

const routeParams = { params: Promise.resolve({ id: "study-1", sourceId: "source-1" }) };

function makeRequest(body: object): NextRequest {
  return new NextRequest("http://localhost:3000/api/studies/study-1/sources/source-1", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function updatedData() {
  return mockPrisma.source.update.mock.calls[0][0].data;
}

describe("PATCH /api/studies/[id]/sources/[sourceId]", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockPrisma.source.findFirst.mockResolvedValue({ id: "source-1", studyId: "study-1" });
    mockPrisma.source.update.mockResolvedValue({ id: "source-1" });
  });

  it("turns a year into a publication date and never passes `year` to Prisma", async () => {
    const response = await PATCH(makeRequest({ title: "Paper", year: "2021" }), routeParams);

    expect(response.status).toBe(200);
    const data = updatedData();
    expect(data).not.toHaveProperty("year");
    expect((data.publicationDate as Date).toISOString()).toBe("2021-12-31T00:00:00.000Z");
  });

  it("drops an empty year, which the metadata editor sends on most saves", async () => {
    const response = await PATCH(makeRequest({ title: "Paper", year: "" }), routeParams);

    expect(response.status).toBe(200);
    const data = updatedData();
    expect(data).not.toHaveProperty("year");
    expect(data.title).toBe("Paper");
  });
});
