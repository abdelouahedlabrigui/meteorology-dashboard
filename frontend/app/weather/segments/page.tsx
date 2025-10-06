import { WeatherSegments } from "@/components/weather-segments"

export default function SegmentsPage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-6 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">Weather Segments Analysis</h1>
          <p className="text-muted-foreground">
            Analyze weather data segmented by time periods with temperature distributions
          </p>
        </div>
        <WeatherSegments />
      </div>
    </div>
  )
}
