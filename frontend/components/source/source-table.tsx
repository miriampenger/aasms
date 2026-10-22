"use client";

import React, { useMemo } from "react";
import Link from "next/link";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getPaginationRowModel,
  flexRender,
  type SortingState,
  type ColumnDef,
} from "@tanstack/react-table";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  RefreshCw,
  FileDown,
  ExternalLink,
  Upload,
  Loader2,
  Check,
  Trash2,
} from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { needsPdfAction, openAccessUrl } from "@/lib/source-pdf-actions";
import type { Source } from "@/types/source";
import {
  getSourceConfidence,
  formatConfidenceTooltip,
  getVotingSortValue,
  compareVotingRank,
} from "@/lib/source-confidence";
import { cn } from "@/lib/utils";

/**
 * A compact icon button with the wording moved into a tooltip.
 *
 * Three labelled buttons no longer fit the actions column, and squeezing them
 * clipped the row; icons keep every action reachable at a fixed width.
 */
function IconAction({ label, children }: { label: string; children: React.ReactElement }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          asChild
          variant="outline"
          size="icon"
          className="h-7 w-7 shrink-0 disabled:opacity-50"
        >
          {children}
        </Button>
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-64">
        {label}
      </TooltipContent>
    </Tooltip>
  );
}

interface SourceTableProps {
  sources: Source[];
  tab: string;
  isEditMode: boolean;
  selectedIds: Set<string>;
  onToggleSelection: (id: string, checked: boolean) => void;
  onDelete: (source: Source) => void;
  suggestions: Record<string, any>;
  loadingMap: Record<string, boolean>;
  onReparse: (id: string) => void;
  onFetchPdf: (id: string) => void;
  onApplySuggestions: (id: string) => void;
  getSourceUrl: (id: string) => string;
  sorting: SortingState;
  onSortingChange: (sorting: SortingState) => void;
}

function SortableHeader({ column, children }: { column: any; children: React.ReactNode }) {
  const sorted = column.getIsSorted();
  return (
    <Button
      variant="ghost"
      size="sm"
      className="-ml-3 h-8"
      onClick={() => column.toggleSorting(sorted === "asc")}
    >
      {children}
      {sorted === "asc" ? (
        <ArrowUp className="ml-1 h-3 w-3" />
      ) : sorted === "desc" ? (
        <ArrowDown className="ml-1 h-3 w-3" />
      ) : (
        <ArrowUpDown className="ml-1 h-3 w-3 opacity-50" />
      )}
    </Button>
  );
}

function CertaintyCell({ source }: { source: Source }) {
  const info = getSourceConfidence(source);

  if (info.confidence === null) {
    return <span className="text-sm text-muted-foreground">—</span>;
  }

  const confidence = `${(info.confidence * 100).toFixed(0)}%`;

  // Without voting there is no agreement to report, so confidence is all there is.
  if (!info.votingEnabled || info.splitDecisions === null) {
    return (
      <span
        className="text-sm tabular-nums text-muted-foreground"
        title={formatConfidenceTooltip(info)}
      >
        {confidence}
      </span>
    );
  }

  const splits = info.splitDecisions;

  return (
    <div className="flex items-center gap-2" title={formatConfidenceTooltip(info)}>
      <Badge
        variant="secondary"
        className={cn(
          "text-xs px-1.5 py-0 font-normal shrink-0 tabular-nums",
          splits === 0
            ? "bg-green-100 text-green-700"
            : splits <= 2
              ? "bg-amber-100 text-amber-700"
              : "bg-red-100 text-red-700",
        )}
      >
        {splits === 0 ? "unanimous" : `${splits} split`}
      </Badge>
      <span className="text-xs tabular-nums text-muted-foreground">{confidence}</span>
    </div>
  );
}

// The trailing column is left unsized so it soaks up whatever width is left
// over, keeping the table flush with its container at any column size.
const LAST_COLUMN_ID = "actions";
// Fits three icon buttons plus delete without the row squeezing them away.
const ACTIONS_MIN_WIDTH = 150;

const STATUS_COLORS: Record<string, string> = {
  PENDING: "bg-gray-100 text-gray-700",
  INCLUDED: "bg-green-100 text-green-700",
  EXCLUDED: "bg-red-100 text-red-700",
  CLASSIFIED: "bg-blue-100 text-blue-700",
  ANALYZING_INCLUSION: "bg-yellow-100 text-yellow-700",
  ANALYZING_CLASSIFICATION: "bg-yellow-100 text-yellow-700",
  NEEDS_REVIEW: "bg-orange-100 text-orange-700",
};

function getDisplayStatus(source: Source): string {
  if (source.finalDecision === "INCLUDE" && source.status === "CLASSIFIED") return "CLASSIFIED";
  if (source.finalDecision === "INCLUDE") return "INCLUDED";
  if (source.finalDecision === "EXCLUDE") return "EXCLUDED";
  return source.status;
}

export function SourceTable({
  sources,
  tab,
  isEditMode,
  selectedIds,
  onToggleSelection,
  onDelete,
  suggestions,
  loadingMap,
  onReparse,
  onFetchPdf,
  onApplySuggestions,
  getSourceUrl,
  sorting,
  onSortingChange,
}: SourceTableProps) {
  const columns = useMemo<ColumnDef<Source>[]>(() => {
    const cols: ColumnDef<Source>[] = [];

    if (isEditMode) {
      cols.push({
        id: "select",
        header: () => null,
        enableResizing: false,
        cell: ({ row }) => (
          <Checkbox
            checked={selectedIds.has(row.original.id)}
            onCheckedChange={(checked) => onToggleSelection(row.original.id, !!checked)}
          />
        ),
        size: 40,
        enableSorting: false,
      });
    }

    cols.push(
      {
        accessorKey: "title",
        size: 420,
        header: ({ column }) => <SortableHeader column={column}>Title</SortableHeader>,
        cell: ({ row }) => (
          <div className="flex items-center gap-2 min-w-0">
            <Link
              href={getSourceUrl(row.original.id)}
              className="font-medium hover:underline truncate min-w-0"
              title={row.original.title}
            >
              {row.original.title}
            </Link>
            {row.original.sourceOrigins?.map((origin) => (
              <Badge key={origin} variant="secondary" className="text-xs shrink-0">
                {origin}
              </Badge>
            ))}
          </div>
        ),
      },
      {
        id: "authors",
        size: 200,
        accessorFn: (row) => row.authors?.join(", ") || "",
        header: ({ column }) => <SortableHeader column={column}>Authors</SortableHeader>,
        cell: ({ getValue }) => (
          <span
            className="truncate block text-sm text-muted-foreground"
            title={getValue() as string}
          >
            {(getValue() as string) || "—"}
          </span>
        ),
      },
      {
        id: "year",
        size: 90,
        accessorFn: (row) =>
          row.publicationDate ? new Date(row.publicationDate).getFullYear() : null,
        header: ({ column }) => <SortableHeader column={column}>Year</SortableHeader>,
        cell: ({ getValue }) => (
          <span className="text-sm">{(getValue() as number | null) ?? "—"}</span>
        ),
        sortingFn: "basic",
      },
      {
        accessorKey: "venue",
        size: 170,
        header: ({ column }) => <SortableHeader column={column}>Venue</SortableHeader>,
        cell: ({ getValue }) => (
          <span
            className="truncate block text-sm text-muted-foreground"
            title={(getValue() as string) || ""}
          >
            {(getValue() as string) || "—"}
          </span>
        ),
      },
      {
        id: "certainty",
        size: 170,
        accessorFn: (row) => getVotingSortValue(getSourceConfidence(row)),
        header: ({ column }) => <SortableHeader column={column}>Certainty</SortableHeader>,
        cell: ({ row }) => <CertaintyCell source={row.original} />,
        // Ascending puts the least agreement — the most contested sources — first.
        sortingFn: (a, b) =>
          compareVotingRank(getSourceConfidence(a.original), getSourceConfidence(b.original)),
        sortUndefined: "last",
      },
      {
        id: "status",
        size: 190,
        accessorFn: (row) => getDisplayStatus(row),
        header: ({ column }) => <SortableHeader column={column}>Status</SortableHeader>,
        cell: ({ row }) => {
          const status = getDisplayStatus(row.original);
          return (
            <Badge
              className={STATUS_COLORS[status] || "bg-gray-100 text-gray-700"}
              variant="secondary"
            >
              {status}
            </Badge>
          );
        },
      },
      {
        id: "actions",
        enableResizing: false,
        header: () => null,
        cell: ({ row }) => {
          const source = row.original;
          const isLoading = loadingMap[source.id];
          return (
            <div className="flex items-center gap-1 justify-end">
              {tab === "pending" && !isEditMode && (
                <>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => onReparse(source.id)}
                    disabled={isLoading}
                    className="h-7 text-xs"
                  >
                    <RefreshCw className="h-3 w-3 mr-1" />
                    {isLoading ? "..." : "Re-parse"}
                  </Button>
                  {suggestions[source.id] && (
                    <Button
                      variant="default"
                      size="sm"
                      onClick={() => onApplySuggestions(source.id)}
                      disabled={isLoading}
                      className="h-7 text-xs"
                    >
                      <Check className="h-3 w-3 mr-1" />
                      Apply
                    </Button>
                  )}
                </>
              )}
              {!isEditMode && needsPdfAction(source) && (
                <>
                  {openAccessUrl(source) && (
                    <IconAction label="Open the paper - it cannot be downloaded automatically, so save the PDF yourself">
                      <a
                        href={openAccessUrl(source)}
                        target="_blank"
                        rel="noopener noreferrer"
                        aria-label="Open the paper"
                      >
                        <ExternalLink className="h-3.5 w-3.5" />
                      </a>
                    </IconAction>
                  )}
                  <IconAction label="Search Unpaywall, OpenAlex, Semantic Scholar and arXiv for a free copy">
                    <button
                      type="button"
                      onClick={() => onFetchPdf(source.id)}
                      disabled={isLoading}
                      aria-label="Find PDF"
                    >
                      {isLoading ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      ) : (
                        <FileDown className="h-3.5 w-3.5" />
                      )}
                    </button>
                  </IconAction>
                  <IconAction label="Upload the PDF yourself">
                    <Link href={getSourceUrl(source.id)} aria-label="Upload PDF">
                      <Upload className="h-3.5 w-3.5" />
                    </Link>
                  </IconAction>
                </>
              )}
              {!isEditMode && (
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-7 w-7 text-muted-foreground hover:text-destructive"
                  onClick={() => onDelete(source)}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              )}
            </div>
          );
        },
        enableSorting: false,
      },
    );

    return cols;
  }, [
    isEditMode,
    selectedIds,
    onToggleSelection,
    tab,
    getSourceUrl,
    suggestions,
    loadingMap,
    onReparse,
    onFetchPdf,
    onApplySuggestions,
    onDelete,
  ]);

  const table = useReactTable({
    data: sources,
    columns,
    state: { sorting },
    enableColumnResizing: true,
    columnResizeMode: "onChange",
    onSortingChange: (updater) => {
      const next = typeof updater === "function" ? updater(sorting) : updater;
      onSortingChange(next);
    },
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    initialState: {
      pagination: { pageSize: 50 },
    },
  });

  if (sources.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No sources found in this category.
      </div>
    );
  }

  const pageIndex = table.getState().pagination.pageIndex;
  const pageSize = table.getState().pagination.pageSize;
  const total = sources.length;
  const pageCount = table.getPageCount();

  return (
    <div className="space-y-4">
      <div className="rounded-md border overflow-x-auto">
        <Table className="table-fixed">
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <TableHead
                    key={header.id}
                    className={cn(
                      "relative py-2 px-3 whitespace-nowrap",
                      header.column.id === LAST_COLUMN_ID ? "" : "overflow-hidden",
                    )}
                    style={
                      header.column.id === LAST_COLUMN_ID
                        ? { minWidth: ACTIONS_MIN_WIDTH }
                        : { width: header.getSize() }
                    }
                  >
                    {header.isPlaceholder
                      ? null
                      : flexRender(header.column.columnDef.header, header.getContext())}
                    {header.column.getCanResize() && (
                      <div
                        role="separator"
                        aria-orientation="vertical"
                        title="Drag to resize, double-click to reset"
                        onMouseDown={header.getResizeHandler()}
                        onTouchStart={header.getResizeHandler()}
                        onDoubleClick={() => header.column.resetSize()}
                        className={cn(
                          "absolute top-0 right-0 h-full w-1.5 cursor-col-resize touch-none select-none",
                          "after:absolute after:inset-y-1 after:right-0 after:w-px after:bg-border after:content-['']",
                          "hover:after:bg-primary hover:after:w-0.5",
                          header.column.getIsResizing() && "after:bg-primary after:w-0.5",
                        )}
                      />
                    )}
                  </TableHead>
                ))}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {table.getRowModel().rows.map((row) => (
              <TableRow key={row.id} className="hover:bg-muted/50">
                {row.getVisibleCells().map((cell) => (
                  <TableCell
                    key={cell.id}
                    className={cn(
                      "py-2 px-3",
                      cell.column.id === LAST_COLUMN_ID ? "whitespace-nowrap" : "overflow-hidden",
                    )}
                    style={
                      cell.column.id === LAST_COLUMN_ID
                        ? { minWidth: ACTIONS_MIN_WIDTH }
                        : { width: cell.column.getSize() }
                    }
                  >
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      {/* Pagination */}
      {total > 25 && (
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">
            Showing {pageIndex * pageSize + 1} to {Math.min((pageIndex + 1) * pageSize, total)} of{" "}
            {total} sources
          </span>
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">Rows per page:</span>
              <Select
                value={pageSize.toString()}
                onValueChange={(v) => {
                  table.setPageSize(parseInt(v, 10));
                }}
              >
                <SelectTrigger className="w-[70px] h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="25">25</SelectItem>
                  <SelectItem value="50">50</SelectItem>
                  <SelectItem value="100">100</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-center gap-1">
              <Button
                variant="outline"
                size="sm"
                onClick={() => table.setPageIndex(0)}
                disabled={!table.getCanPreviousPage()}
              >
                <ChevronsLeft className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => table.previousPage()}
                disabled={!table.getCanPreviousPage()}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <span className="text-sm px-2">
                Page {pageIndex + 1} of {pageCount}
              </span>
              <Button
                variant="outline"
                size="sm"
                onClick={() => table.nextPage()}
                disabled={!table.getCanNextPage()}
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => table.setPageIndex(pageCount - 1)}
                disabled={!table.getCanNextPage()}
              >
                <ChevronsRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
