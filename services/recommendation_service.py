import grpc
import recommendation_pb2
import recommendation_pb2_grpc
import pickle
import pandas as pd
from concurrent import futures

# Load trained model
with open("models/recommender.pkl", "rb") as f:
    model = pickle.load(f)

# Sample user-item data (Make sure it matches the trained model)
user_item_matrix = pd.DataFrame({'A': [1, 0, 1], 'B': [0, 1, 0], 'C': [0, 1, 1]})

class RecommendationService(recommendation_pb2_grpc.RecommendationServiceServicer):
    def GetRecommendations(self, request, context):
        try:
            # Get user ID (convert to integer)
            user_id = int(request.user_id)

            # Validate user ID
            if user_id >= len(user_item_matrix):
                return recommendation_pb2.RecommendationResponse(recommended_items=[])

            # Get user vector and reshape
            user_vector = pd.DataFrame([user_item_matrix.iloc[user_id]], columns=user_item_matrix.columns)

            # Get recommendations using k-nearest neighbors
            _, indices = model.kneighbors(user_vector, n_neighbors=2)

            # Convert indices to column names
            recommended_items = [user_item_matrix.columns[i] for i in indices.flatten()]

            return recommendation_pb2.RecommendationResponse(recommended_items=recommended_items)

        except Exception as e:
            print(f"Error in recommendation: {e}")
            return recommendation_pb2.RecommendationResponse(recommended_items=[])

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    recommendation_pb2_grpc.add_RecommendationServiceServicer_to_server(RecommendationService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("✅ Recommendation Service running on port 50051")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
