package network;

import retrofit2.Call;
import retrofit2.http.GET;

import java.util.List;

public interface GraphicsAPI {
    @GET("objects")
    Call<List<GraphicalObjectDTO>> fetchObject();
}
