from fastapi import FastAPI
import grpc
import recommendation_pb2
import recommendation_pb2_grpc

app = FastAPI()

# Connect to gRPC server
channel = grpc.insecure_channel("localhost:50051")
stub = recommendation_pb2_grpc.RecommendationServiceStub(channel)

@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: str):
    response = stub.GetRecommendations(recommendation_pb2.UserRequest(user_id=user_id))
    return {"user_id": user_id, "recommended_items": response.recommended_items}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
