import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp_and_remote():
    """
    Simulates an MCP client connecting to our remote_api_mcp server over stdio,
    then calling a remote tool.
    """
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server/remote_api_mcp.py"],
        env={"PYTHONPATH": "."}
    )

    print("--- Initializing MCP Session ---")
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # 1. List available tools
                print("\nListing available tools:")
                tools_response = await session.list_tools()
                for tool in tools_response.tools:
                    print(f"- {tool.name}: {tool.description[:50]}...")

                # 2. Call predict_loan
                print("\n--- Testing 'predict_loan' through MCP ---")
                loan_payload = {
                    "gender": "Male",
                    "married": "Yes",
                    "dependents": "0",
                    "education": "Graduate",
                    "self_employed": "No",
                    "applicant_income": 5821.0,
                    "coapplicant_income": 0.0,
                    "loan_amount": 144.0,
                    "loan_amount_term": 360.0,
                    "credit_history": 1.0,
                    "property_area": "Urban"
                }
                loan_result = await session.call_tool("predict_loan", arguments=loan_payload)
                print(f"Loan Prediction Result: {loan_result.content[0].text}")

                # 3. Call predict_heart_disease
                print("\n--- Testing 'predict_heart_disease' through MCP ---")
                heart_payload = {
                    "age": 63.0,
                    "sex": 1,
                    "cp": 3,
                    "trestbps": 145.0,
                    "chol": 233.0,
                    "fbs": 1,
                    "restecg": 0,
                    "thalach": 150.0,
                    "exang": 0,
                    "oldpeak": 2.3,
                    "slope": 0,
                    "ca": 0,
                    "thal": 1
                }
                heart_result = await session.call_tool("predict_heart_disease", arguments=heart_payload)
                print(f"Heart Disease Prediction Result: {heart_result.content[0].text}")

                # 4. Call predict_stock_price
                print("\n--- Testing 'predict_stock_price' through MCP ---")
                # Sample 20 days of data
                stock_data = []
                for i in range(20):
                    stock_data.append({"open": 150.0+i, "high": 155.0+i, "low": 149.0+i, "close": 152.0+i, "volume": 1000000.0, "average": 152.0+i})
                
                stock_payload = {"stock_data": stock_data}
                stock_result = await session.call_tool("predict_stock_price", arguments=stock_payload)
                print(f"Stock Price Prediction Result: {stock_result.content[0].text}")
                
    except Exception as e:
        print(f"Error during MCP interaction: {e}")

if __name__ == "__main__":
    asyncio.run(test_mcp_and_remote())
