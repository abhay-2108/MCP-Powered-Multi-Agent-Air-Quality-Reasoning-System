import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run_orchestrator_scenario():
    server_params = StdioServerParameters(
        command="python",
        args=["p:\\College Projects\\Deep Learning\\MCP-Powered Multi-Agent Air Quality Reasoning System\\mcp_server\\server.py"],
        env=None
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("\n--- Air Quality Risk Intelligence Agent ---")
            print("Scenario: High traffic area with visible haze detected.\n")

            image_path = "p:\\College Projects\\Deep Learning\\MCP-Powered Multi-Agent Air Quality Reasoning System\\Images\\polluted_sample.jpg" 
            print(f"[Agent] Calling Visual Classifier for: {image_path}")
            vision_result = await session.call_tool("classify_pollution_image", {"image_path": image_path})
            print(f"[Result] {vision_result.content[0].text}")

            print("[Agent] Analyzing Traffic Emissions...")
            emission_result = await session.call_tool("predict_vehicle_emission", {
                "engine_size": 3.5, "mileage": 120000, "speed": 45.0
            })
            print(f"[Result] {emission_result.content[0].text}")

            print("[Agent] Generating 4-Hour AQI Forecast...")
            forecast_result = await session.call_tool("predict_aqi_forecast", {
                "pm25": 85.0, "pm10": 150.0, "no2": 45.0, "so2": 12.0, "o3": 65.0, "co": 2.5
            })
            print(f"[Result] {forecast_result.content[0].text}")

            print("[Agent] Evaluating Public Health Risk...")
            health_result = await session.call_tool("estimate_health_risk", {
                "aqi": 160.0, "pm25": 85.0, "temperature": 32.0, "humidity": 65.0
            })
            print(f"[Result] {health_result.content[0].text}")

            print("\n--- Final Agent Summary ---")
            print("The system has identified high emissions from vehicular traffic as a primary source of current haze.")
            print("The 4-hour forecast suggests sustained 'Moderate' to 'Poor' air quality.")
            print("Advice: Sensitive groups should avoid outdoor exertion. High-efficiency air filters recommended for nearby residences.")

if __name__ == "__main__":
    asyncio.run(run_orchestrator_scenario())
