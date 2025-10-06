"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Search, MapPin, ChevronLeft, ChevronRight, Home } from "lucide-react"
import Link from "next/link"

interface ZipData {
  zip: string
  city: string
  state: string
  state_abbr: string
  county: string
  count_code: string
  latitude: string
  longitude: string
}

const API_BASE = "http://10.42.0.243:8003"

export function ZipCodeSearch() {
  const [city, setCity] = useState("")
  const [state, setState] = useState("")
  const [county, setCounty] = useState("")
  const [results, setResults] = useState<ZipData[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [currentPage, setCurrentPage] = useState(1)
  const itemsPerPage = 10

  const handleSearch = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE}/us/zips`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ city, state, county }),
      })

      if (!response.ok) throw new Error("Failed to fetch ZIP codes")

      const data = await response.json()
      setResults(data)
      setCurrentPage(1)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
    } finally {
      setLoading(false)
    }
  }

  const totalPages = Math.ceil(results.length / itemsPerPage)
  const startIndex = (currentPage - 1) * itemsPerPage
  const paginatedResults = results.slice(startIndex, startIndex + itemsPerPage)

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <MapPin className="h-6 w-6 text-primary" />
            <h1 className="text-2xl font-bold text-foreground">ZIP Code Lookup</h1>
          </div>
          <Link href="/">
            <Button variant="ghost" size="sm">
              <Home className="h-4 w-4 mr-2" />
              Home
            </Button>
          </Link>
        </div>
      </header>

      <main className="container mx-auto px-6 py-8">
        <Card>
          <CardHeader>
            <CardTitle>Search ZIP Codes</CardTitle>
            <CardDescription>Find ZIP codes by city, state, and county</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div className="space-y-2">
                <Label htmlFor="city">City</Label>
                <Input id="city" placeholder="e.g., Tampa" value={city} onChange={(e) => setCity(e.target.value)} />
              </div>
              <div className="space-y-2">
                <Label htmlFor="state">State</Label>
                <Input
                  id="state"
                  placeholder="e.g., Florida"
                  value={state}
                  onChange={(e) => setState(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="county">County</Label>
                <Input
                  id="county"
                  placeholder="e.g., Hillsborough"
                  value={county}
                  onChange={(e) => setCounty(e.target.value)}
                />
              </div>
            </div>

            <Button onClick={handleSearch} disabled={loading} className="w-full md:w-auto">
              <Search className="h-4 w-4 mr-2" />
              {loading ? "Searching..." : "Search"}
            </Button>

            {error && (
              <div className="mt-4 p-4 bg-destructive/10 border border-destructive rounded-md">
                <p className="text-sm text-destructive">{error}</p>
              </div>
            )}

            {results.length > 0 && (
              <div className="mt-6">
                <div className="rounded-md border border-border">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>ZIP</TableHead>
                        <TableHead>City</TableHead>
                        <TableHead>State</TableHead>
                        <TableHead>County</TableHead>
                        <TableHead>Latitude</TableHead>
                        <TableHead>Longitude</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {paginatedResults.map((zip, index) => (
                        <TableRow key={`${zip.zip}-${index}`}>
                          <TableCell className="font-medium">{zip.zip}</TableCell>
                          <TableCell>{zip.city}</TableCell>
                          <TableCell>
                            {zip.state} ({zip.state_abbr})
                          </TableCell>
                          <TableCell>{zip.county}</TableCell>
                          <TableCell>{zip.latitude}</TableCell>
                          <TableCell>{zip.longitude}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>

                {totalPages > 1 && (
                  <div className="flex items-center justify-between mt-4">
                    <p className="text-sm text-muted-foreground">
                      Showing {startIndex + 1} to {Math.min(startIndex + itemsPerPage, results.length)} of{" "}
                      {results.length} results
                    </p>
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                        disabled={currentPage === 1}
                      >
                        <ChevronLeft className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                        disabled={currentPage === totalPages}
                      >
                        <ChevronRight className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </main>
    </div>
  )
}
