"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Loader2 } from "lucide-react"
import ReactMarkdown from 'react-markdown';


const API_BASE = "http://10.42.0.243:8003"

interface SegmentData {
  time_range: string
  temperature_2m: { max: number; min: number; avg: number }
  relative_humidity_2m: { max: number; min: number; avg: number }
  wind_speed_10m: { max: number; min: number; avg: number }
  precipitation: { max: number; min: number; total: number }
  weather_code: { most_common: number }
  pressure_msl: { max: number; min: number; avg: number }
  cloud_cover: { max: number; min: number; avg: number }
}

interface SegmentResult {
  original_data: {
    latitude: string
    longitude: string
    city: string
    state: string
    county: string
    start_date: string
    end_date: string
    segmented_data: SegmentData[]
  }
  temperature_analysis: {
    celsius_parameters: { mu: number; sigma: number }
    fahrenheit_parameters: { mu: number; sigma: number }
    plot: string
    plot_description: string
    sample_size: number
    temperature_range_c: { min: number; max: number }
  }
}

export function WeatherSegments() {
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<SegmentResult[] | null>(null)
  const [postFormData, setPostFormData] = useState({
    month: "june",
  })
  const [getFormData, setGetFormData] = useState({
    lat: "",
    lon: "",
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    try {
      const response = await fetch(`${API_BASE}/weather/segments`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(postFormData),
      })
      const data = await response.json()
      console.log("[v0] Segments POST response:", data)
      setResults(data.results || [data])
    } catch (error) {
      console.error("[v0] Error fetching segments:", error)
    } finally {
      setLoading(false)
    }
  }

  const handleGetSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    try {
      const response = await fetch(`${API_BASE}/search/segments?lat=${getFormData.lat}&lon=${getFormData.lon}`)
      const data = await response.json()
      console.log("[v0] Segments GET response:", data)
      setResults(data.results || [data])
    } catch (error) {
      console.error("[v0] Error searching segments:", error)
    } finally {
      setLoading(false)
    }
  }

  const isCode = (text: string) => {
    const codePatterns = [
      /```[\s\S]*```/, // Code blocks
      /`[^`]+`/, // Inline code
      /^\s*(?:function|class|def|import|from|const|let|var|if|for|while)\s/m, // Keywords
      /[{}();]/, // Common code symbols
      /^\s*\/\/|^\s*#|^\s*\/\*/m, // Comments
    ]
    return codePatterns.some((pattern) => pattern.test(text))
  }

  const formatContent = (text: string) => {
    // Handle markdown code blocks
    const codeBlockRegex = /```(\w+)?\n?([\s\S]*?)```/g
    const parts = []
    let lastIndex = 0
    let match

    while ((match = codeBlockRegex.exec(text)) !== null) {
      // Add text before code block
      if (match.index > lastIndex) {
        const beforeText = text.slice(lastIndex, match.index)
        if (beforeText.trim()) {
          parts.push({
            type: "text",
            content: beforeText.trim(),
            key: `text-${lastIndex}`,
          })
        }
      }

      // Add code block
      parts.push({
        type: "code",
        language: match[1] || "text",
        content: match[2].trim(),
        key: `code-${match.index}`,
      })

      lastIndex = match.index + match[0].length
    }

    // Add remaining text
    if (lastIndex < text.length) {
      const remainingText = text.slice(lastIndex)
      if (remainingText.trim()) {
        parts.push({
          type: "text",
          content: remainingText.trim(),
          key: `text-${lastIndex}`,
        })
      }
    }

    // If no code blocks found, treat as single piece
    if (parts.length === 0) {
      parts.push({
        type: isCode(text) ? "code" : "text",
        content: text,
        language: "text",
        key: "single",
      })
    }

    return parts
  }

  return (
    <div className="space-y-6">
      <Tabs defaultValue="post" className="w-full">
        <TabsList className="grid w-full max-w-md grid-cols-2">
          <TabsTrigger value="post">Create Analysis</TabsTrigger>
          <TabsTrigger value="get">Search by Location</TabsTrigger>
        </TabsList>

        <TabsContent value="post" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Create Segment Analysis</CardTitle>
              <CardDescription>Enter month to analyze weather segments</CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="month">Month</Label>
                  <Input
                    id="month"
                    value={postFormData.month}
                    onChange={(e) => setPostFormData({ month: e.target.value })}
                    placeholder="june"
                    required
                  />
                </div>
                <Button type="submit" disabled={loading} className="w-full">
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    "Analyze Segments"
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="get" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Search Segment Analyses</CardTitle>
              <CardDescription>Search for segment analyses by latitude and longitude</CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleGetSearch} className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="lat">Latitude</Label>
                    <Input
                      id="lat"
                      value={getFormData.lat}
                      onChange={(e) => setGetFormData({ ...getFormData, lat: e.target.value })}
                      placeholder="45.5235"
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="lon">Longitude</Label>
                    <Input
                      id="lon"
                      value={getFormData.lon}
                      onChange={(e) => setGetFormData({ ...getFormData, lon: e.target.value })}
                      placeholder="-122.6762"
                      required
                    />
                  </div>
                </div>
                <Button type="submit" disabled={loading} className="w-full">
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Searching...
                    </>
                  ) : (
                    "Search Analyses"
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {results && results.length > 0 && (
        <div className="space-y-6">
          {results.map((result, idx) => (
            <div key={idx} className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Location Information</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div>
                      <p className="text-sm text-muted-foreground">City</p>
                      <p className="font-medium">{result.original_data.city}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">State</p>
                      <p className="font-medium">{result.original_data.state}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Latitude</p>
                      <p className="font-medium">{result.original_data.latitude}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Longitude</p>
                      <p className="font-medium">{result.original_data.longitude}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Start Date</p>
                      <p className="font-medium">{result.original_data.start_date}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">End Date</p>
                      <p className="font-medium">{result.original_data.end_date}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Temperature Analysis</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div>
                      <p className="text-sm text-muted-foreground">Mean (°C)</p>
                      <p className="text-2xl font-bold text-primary">
                        {result.temperature_analysis.celsius_parameters.mu.toFixed(2)}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Std Dev (°C)</p>
                      <p className="text-2xl font-bold">
                        {result.temperature_analysis.celsius_parameters.sigma.toFixed(2)}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Min (°C)</p>
                      <p className="text-2xl font-bold">
                        {result.temperature_analysis.temperature_range_c.min.toFixed(2)}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Max (°C)</p>
                      <p className="text-2xl font-bold">
                        {result.temperature_analysis.temperature_range_c.max.toFixed(2)}
                      </p>
                    </div>
                  </div>
                  {result.temperature_analysis.plot && (
                    <div className="space-y-2">
                      <h4 className="font-semibold">Temperature Distribution</h4>
                      <img
                        src={`data:image/png;base64,${result.temperature_analysis.plot}`}
                        alt="Temperature Distribution"
                        className="w-full rounded-lg border border-border"
                      />
                      <p className="text-sm text-muted-foreground">
                          <ReactMarkdown>{result.temperature_analysis.plot_description}</ReactMarkdown>

                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Weather Segments ({result.original_data.segmented_data.length})</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {result.original_data.segmented_data.map((segment, segIdx) => (
                      <Card key={segIdx} className="bg-accent/50">
                        <CardHeader>
                          <CardTitle className="text-base">{segment.time_range}</CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                            <div>
                              <p className="text-muted-foreground">Temperature</p>
                              <p className="font-medium">
                                {segment.temperature_2m.avg.toFixed(1)}°C ({segment.temperature_2m.min.toFixed(1)} -{" "}
                                {segment.temperature_2m.max.toFixed(1)})
                              </p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Humidity</p>
                              <p className="font-medium">
                                {segment.relative_humidity_2m.avg.toFixed(0)}% ({segment.relative_humidity_2m.min} -{" "}
                                {segment.relative_humidity_2m.max})
                              </p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Wind Speed</p>
                              <p className="font-medium">
                                {segment.wind_speed_10m.avg.toFixed(1)} km/h ({segment.wind_speed_10m.min.toFixed(1)} -{" "}
                                {segment.wind_speed_10m.max.toFixed(1)})
                              </p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Precipitation</p>
                              <p className="font-medium">{segment.precipitation.total.toFixed(1)} mm</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Pressure</p>
                              <p className="font-medium">{segment.pressure_msl.avg.toFixed(1)} hPa</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Cloud Cover</p>
                              <p className="font-medium">{segment.cloud_cover.avg.toFixed(0)}%</p>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
