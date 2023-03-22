package network;

import objects.GraphicalObject;
import org.json.JSONArray;
import org.json.JSONObject;
import utils.NetworkCommands;
import utils.ObjectInfo;
import utils.network_events.*;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;

public class TCPNetworkRepository implements NetworkRepository, Runnable {
    public static final int SERVER_PORT = 8080;
    public static final String HOST = "localhost";
    private ServerSocket serverSocket;
    private Socket clientSocket;

    private final NetworkEventListener listener;

    private PrintWriter out;
    private BufferedReader in;

    public TCPNetworkRepository(boolean isServer, NetworkEventListener listener) {
        this.listener = listener;
        InputStream inputStream;
        OutputStream outputStream;
        if (isServer) {
            try {
                serverSocket = new ServerSocket(SERVER_PORT);
                System.out.println("waiting for client...");
                clientSocket = serverSocket.accept();
                inputStream = clientSocket.getInputStream();
                outputStream = clientSocket.getOutputStream();
                out = new PrintWriter(outputStream, true);
                in = new BufferedReader(new InputStreamReader(inputStream));
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            try {
                clientSocket = new Socket(HOST, SERVER_PORT);
                inputStream = clientSocket.getInputStream();
                outputStream = clientSocket.getOutputStream();
                out = new PrintWriter(outputStream, true);
                in = new BufferedReader(new InputStreamReader(inputStream));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public void closeConnection() {
        try {
            if (serverSocket != null) {
                serverSocket.close();
            }
            if (clientSocket != null) {
                clientSocket.close();
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void clearObjects() {
        JSONObject jsonObject = new JSONObject();
        jsonObject.put("command", NetworkCommands.CLEAR_OBJECTS);
        var res = jsonObject.toString();
        out.println(res);
    }

    @Override
    public void sendObject(GraphicalObject object) {
        var jsonObject = new JSONObject();
        jsonObject.put("command", NetworkCommands.RESPONSE_OBJ);
        var type = object.getClass().getSimpleName();
        jsonObject.put("obj_type", type);
        jsonObject.put("object", object.writeToJson());
        var res = jsonObject.toString();
        out.println(res);
    }

    @Override
    public void sendObjectByIndex(int index, GraphicalObject object) {
        var jsonObject = new JSONObject();
        jsonObject.put("command", NetworkCommands.RESPONSE_OBJ_BY_INDEX);
        jsonObject.put("object", object.writeToJson());
        var type = object.getClass().getSimpleName();
        jsonObject.put("obj_type", type);
        jsonObject.put("index", index);
        var res = jsonObject.toString();
        out.println(res);
    }

    @Override
    public void sendObjectsList(ObjectInfo[] objects) {
        var jsonObject = new JSONObject();
        jsonObject.put("command", NetworkCommands.RESPONSE_OBJ_LIST);
        var jsonArray = new JSONArray();
        for (ObjectInfo object : objects) {
            var obj = new JSONObject();
            obj.put("type", object.objectType());
            obj.put("hash", object.hash());
            jsonArray.put(obj);
        }
        jsonObject.put("objects", jsonArray);
        var res = jsonObject.toString();
        out.println(res);
    }

    @Override
    public void sendObjectsListSize(int size) {
        var jsonObject = new JSONObject();
        jsonObject.put("command", NetworkCommands.RESPONSE_OBJ_LIST_SIZE);
        jsonObject.put("obj_list_size", size);
        var res = jsonObject.toString();
        out.println(res);
    }

    @Override
    public void requestObjectByIndex(int index) {
        var jsonObject = new JSONObject();
        jsonObject.put("command", NetworkCommands.REQUEST_OBJ_BY_INDEX);
        jsonObject.put("index", index);
        var res = jsonObject.toString();
        out.println(res);
    }

    @Override
    public void requestObjectsList() {
        var jsonObject = new JSONObject();
        jsonObject.put("command", NetworkCommands.REQUEST_OBJ_LIST);
        var res = jsonObject.toString();
        out.println(res);
    }

    @Override
    public void requestObjectsListSize() {
        var jsonObject = new JSONObject();
        jsonObject.put("command", NetworkCommands.REQUEST_OBJ_LIST_SIZE);
        var res = jsonObject.toString();
        out.println(res);
    }

    @Override
    public void run() {
        eventLoop:
        while (true) {
            try {
                String line = in.readLine();
                System.out.println("received: " + line);
                var jsonObject = new JSONObject(line);
                var command = jsonObject.getString("command");
                switch (command) {
                    case NetworkCommands.CLOSE_CONNECTION -> {
                        listener.onEvent(new CloseConnection());
                        closeConnection();
                        break eventLoop;
                    }
                    case NetworkCommands.CLEAR_OBJECTS -> listener.onEvent(new ClearObjects());
                    case NetworkCommands.REQUEST_OBJ_BY_INDEX -> {
                        var index = jsonObject.getInt("index");
                        listener.onEvent(new RequestObjectByIndex(index));
                    }
                    case NetworkCommands.REQUEST_OBJ_LIST -> listener.onEvent(new RequestObjectList());
                    case NetworkCommands.REQUEST_OBJ_LIST_SIZE -> listener.onEvent(new RequestObjectListSize());
                    case NetworkCommands.RESPONSE_OBJ_BY_INDEX ->
                            listener.onEvent(new ResponseObjectByIndex(jsonObject.getInt("index"), jsonObject.getString("obj_type"), jsonObject.getString("object")));
                    case NetworkCommands.RESPONSE_OBJ_LIST -> {
                        var jsonArray = jsonObject.getJSONArray("objects");
                        var objects = new ObjectInfo[jsonArray.length()];
                        for (int i = 0; i < jsonArray.length(); i++) {
                            var obj = jsonArray.getJSONObject(i);
                            objects[i] = new ObjectInfo(obj.getString("type"), obj.getString("hash"));
                        }
                        listener.onEvent(new ResponseObjectList(objects));
                    }
                    case NetworkCommands.RESPONSE_OBJ_LIST_SIZE ->
                            listener.onEvent(new ResponseObjectListSize(jsonObject.getInt("obj_list_size")));
                    case NetworkCommands.RESPONSE_OBJ ->
                            listener.onEvent(new ResponseObject(jsonObject.getString("object")));
                }
            } catch (IOException e) {
                System.out.println(e.getMessage());
            }
        }
    }
}
