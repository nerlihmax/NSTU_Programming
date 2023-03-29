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
import utils.network_events.ResponseObject;
import utils.network_events.ResponseObjectList;
import utils.network_events.ResponseObjectListNames;

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
    public void sendObject(GraphicalObject object) {
        api.sendObject(GraphicalObjectDTO.fromGraphicalObject(object)).enqueue(new Callback<>() {
            @Override
            public void onResponse(Call<Void> call, Response<Void> response) {
                System.out.println("Object sent");
            }

            @Override
            public void onFailure(Call<Void> call, Throwable throwable) {
                System.out.println("Error occurred");
                throwable.printStackTrace();
            }
        });
    }

    @Override
    public void requestObjectByIndex(int index) {
        api.fetchObjectByIndex(index).enqueue(new Callback<>() {
            @Override
            public void onResponse(Call<GraphicalObjectDTO> call, Response<GraphicalObjectDTO> response) {
                if (response.body() == null) {
                    System.out.println("Body is empty");
                    return;
                }
                listener.onEvent(new ResponseObject(response.body().toGraphicalObject()));
            }

            @Override
            public void onFailure(Call<GraphicalObjectDTO> call, Throwable throwable) {
                System.out.println("Error occurred");
                throwable.printStackTrace();
            }
        });
    }

    @Override
    public void requestObjectsList() {
        api.fetchObjects().enqueue(new Callback<>() {
            @Override
            public void onResponse(Call<List<GraphicalObjectDTO>> call, Response<List<GraphicalObjectDTO>> response) {
                if (response.body() == null) {
                    System.out.println("Body is empty");
                    return;
                }
                var list = response.body().stream().map(GraphicalObjectDTO::toGraphicalObject).toArray(GraphicalObject[]::new);
                listener.onEvent(new ResponseObjectList(list));
            }

            @Override
            public void onFailure(Call<List<GraphicalObjectDTO>> call, Throwable throwable) {
                System.out.println("Error occurred");
                throwable.printStackTrace();
            }
        });
    }

    @Override
    public void requestObjectsListNames() {
        api.fetchObjects().enqueue(new Callback<>() {
            @Override
            public void onResponse(Call<List<GraphicalObjectDTO>> call, Response<List<GraphicalObjectDTO>> response) {
                if (response.body() == null) {
                    System.out.println("Body is empty");
                    return;
                }
                var list = response.body().stream().map(GraphicalObjectDTO::toGraphicalObject).toArray(GraphicalObject[]::new);
                listener.onEvent(new ResponseObjectListNames(list));
            }

            @Override
            public void onFailure(Call<List<GraphicalObjectDTO>> call, Throwable throwable) {
                System.out.println("Error occurred");
                throwable.printStackTrace();
            }
        });
    }

    @Override
    public void removeObjectByIndex(int index) {
        api.removeObjectByIndex(index).enqueue(new Callback<>() {
            @Override
            public void onResponse(Call<Void> call, Response<Void> response) {
                System.out.println("Object removed");
            }

            @Override
            public void onFailure(Call<Void> call, Throwable throwable) {
                System.out.println("Error occurred");
                throwable.printStackTrace();
            }
        });
    }
}
