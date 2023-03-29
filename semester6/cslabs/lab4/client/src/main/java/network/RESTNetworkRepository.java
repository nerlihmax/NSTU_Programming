package network;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import objects.GraphicalObject;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;
import utils.RuntimeTypeAdapterFactory;
import utils.network_events.ResponseObjectList;

import javax.xml.transform.Source;
import java.util.List;

public class RESTNetworkRepository implements NetworkRepository {

    private final NetworkEventListener listener;

    static final String BASE_URL = "http://localhost:4567/";

    private GraphicsAPI api;

    public void start() {
        RuntimeTypeAdapterFactory<GraphicalObjectDTO> runtimeTypeAdapterFactory
                = RuntimeTypeAdapterFactory
                .of(GraphicalObjectDTO.class, "type")
                .registerSubtype(StarDTO.class, "star")
                .registerSubtype(SmileyDTO.class, "smiley");
        Gson gson = new GsonBuilder()
                .setLenient()
                .registerTypeAdapterFactory(runtimeTypeAdapterFactory)
                .create();
        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl(BASE_URL)
                .addConverterFactory(GsonConverterFactory.create(gson))
                .build();
        api = retrofit.create(GraphicsAPI.class);
    }

    public RESTNetworkRepository(NetworkEventListener listener) {
        this.listener = listener;
        start();
    }

    @Override
    public void closeConnection() {
    }

    @Override
    public void clearObjects() {
    }

    @Override
    public void sendObjectByIndex(int index, GraphicalObject object) {

    }

    @Override
    public void sendObjectsList(GraphicalObject[] objects) {

    }

    @Override
    public void sendObjectsListSize(int size) {

    }

    @Override
    public void requestObjectByIndex(int index) {

    }

    @Override
    public void requestObjectsList() {
        api.fetchObject().enqueue(new Callback<>() {
            @Override
            public void onResponse(Call<List<GraphicalObjectDTO>> call, Response<List<GraphicalObjectDTO>> response) {
                if (response.body() == null) {
                    System.out.println("Body is empty");
                    return;
                }
                response.body().forEach(System.out::println);
                System.out.println(response.body().toString());
            }

            @Override
            public void onFailure(Call<List<GraphicalObjectDTO>> call, Throwable throwable) {
                System.out.println("Error occurred");
                throwable.printStackTrace();
            }
        });
    }

    @Override
    public void requestObjectsListSize() {

    }
}
