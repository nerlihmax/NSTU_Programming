package network;

import retrofit2.Call;
import retrofit2.http.*;

import java.util.List;

public interface GraphicsAPI {
    @GET("objects")
    Call<List<GraphicalObjectDTO>> fetchObjects();

    @GET("object/get")
    Call<GraphicalObjectDTO> fetchObjectByIndex(@Query("id") int index);

    @POST("object/add")
    Call<Void> sendObject(@Body GraphicalObjectDTO object);

    @DELETE("object/remove")
    Call<Void> removeObjectByIndex(@Query("id") int index);
}
