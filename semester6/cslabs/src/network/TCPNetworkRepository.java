package network;

import objects.GraphicalObject;
import org.json.JSONArray;
import org.json.JSONObject;
import utils.NetworkCommands;
import utils.ObjectInfo;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.List;

public class TCPNetworkRepository implements NetworkRepository {
    public static final int SERVER_PORT = 8080;
    public static final String HOST = "localhost";
    private ServerSocket serverSocket = null;
    private Socket clientSocket = null;

    private InputStream inputStream;
    private OutputStream outputStream;

    public TCPNetworkRepository(boolean isServer) {
        if (isServer) {
            try {
                serverSocket = new ServerSocket(SERVER_PORT);
                System.out.println("waiting for client...");
                clientSocket = serverSocket.accept();
                inputStream = clientSocket.getInputStream();
                outputStream = clientSocket.getOutputStream();
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            try {
                clientSocket = new Socket(HOST, SERVER_PORT);
                inputStream = clientSocket.getInputStream();
                outputStream = clientSocket.getOutputStream();
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
        jsonObject.put("command", "clear");
        try {
            outputStream.write(jsonObject.toString().getBytes());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void sendObject(GraphicalObject object) {
        var jsonObject = new JSONObject();
        jsonObject.put("command", NetworkCommands.GET_OBJ);
        try {
            jsonObject.put("object", object.write());
            outputStream.write(jsonObject.toString().getBytes());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void sendObjectsList(List<ObjectInfo> objects) {
        var jsonObject = new JSONObject();
        jsonObject.put("command", NetworkCommands.GET_OBJ_LIST);
        try {
            var jsonArray = new JSONArray(objects);
            jsonObject.put("object", jsonArray);
            outputStream.write(jsonObject.toString().getBytes());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void sendObjectsListSize(int size) {
        var jsonObject = new JSONObject();
        jsonObject.put("command", NetworkCommands.GET_OBJ_LIST_SIZE);
        try {
            jsonObject.put("obj_list_size", size);
            outputStream.write(jsonObject.toString().getBytes());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public InputStream getInputStream() {
        return inputStream;
    }
}
