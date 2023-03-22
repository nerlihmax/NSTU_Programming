import network.NetworkEventListener;
import network.NetworkRepository;
import network.TCPNetworkRepository;
import objects.Smiley;
import objects.Star;
import utils.ObjectInfo;
import utils.network_events.*;

import java.awt.*;
import java.util.Arrays;
import java.util.Scanner;

class Util {
    static void handleEvent(NetworkEvent event, NetworkRepository networkRepository) {
        switch (event) {
            case RequestObjectByIndex requestObjectByIndex -> {
                System.out.println("RequestObjectByIndex" + requestObjectByIndex.index());
                networkRepository.sendObjectByIndex(requestObjectByIndex.index(), new Star(100, 100, 100, 100, Color.RED));
            }
            case RequestObjectList ignored -> {
                System.out.println("RequestObjectList");
                networkRepository.sendObjectsList(new ObjectInfo[]{new ObjectInfo("Star", "1231234"), new ObjectInfo("Smiley", "452345236")});
            }
            case ClearObjects ignored -> System.out.println("ClearObjects");
            case ResponseObjectByIndex responseObjectByIndex -> {
                System.out.println("ResponseObjectByIndex: " + "index: " + responseObjectByIndex.index() + " object: " + responseObjectByIndex.object());
                var object = switch (responseObjectByIndex.objectType()) {
                    case "Star" -> new Star(100, 100, 100, 100, Color.RED);
                    case "Smiley" -> new Smiley(100, 100, 100, 100, Color.RED);
                    default ->
                            throw new IllegalStateException("Unexpected value: " + responseObjectByIndex.objectType());
                };
                object.readFromJson(responseObjectByIndex.object());
                System.out.println(object.writeToJson());
            }
            case CloseConnection ignored -> System.out.println("CloseConnection");
            case RequestObjectListSize ignored -> {
                System.out.println("RequestObjectListSize");
                networkRepository.sendObjectsListSize(2);
            }
            case ResponseObjectList responseObjectList -> System.out.println("ResponseObjectList: " + Arrays.toString(responseObjectList.objects()));
            case ResponseObjectListSize responseObjectListSize ->
                    System.out.println("ResponseObjectListSize: " + responseObjectListSize.size());
            case ResponseObject responseObject -> System.out.println("ResponseObject: " + responseObject.object());
        }
    }
}

class ServerTest implements NetworkEventListener {
    NetworkRepository networkRepository;

    public static void main(String[] args) {
        new ServerTest();
    }

    public ServerTest() {
        super();
        networkRepository = new TCPNetworkRepository(true, this);
        var thread = new Thread((Runnable) networkRepository);
        thread.setDaemon(true);
        thread.start();

        Scanner scanner = new Scanner(System.in);
        String input;
        boolean running = true;

        while (running) {
            System.out.println("Please select an option:");
            System.out.println("1. CLOSE_CONNECTION");
            System.out.println("2. CLEAR_OBJECTS");
            System.out.println("3. REQUEST_OBJ_BY_INDEX");
            System.out.println("4. REQUEST_OBJ_LIST");
            System.out.println("5. REQUEST_OBJ_LIST_SIZE");
            System.out.println("0. Exit");

            input = scanner.nextLine();

            switch (input) {
                case "1" -> networkRepository.closeConnection();
                case "2" -> networkRepository.clearObjects();
                case "3" -> networkRepository.requestObjectByIndex(1);
                case "4" -> networkRepository.requestObjectsList();
                case "5" -> networkRepository.requestObjectsListSize();
                case "0" -> running = false;
                default -> System.out.println("Invalid input, please try again.");
            }
        }
        scanner.close();
    }

    @Override
    public void onEvent(NetworkEvent event) {
        Util.handleEvent(event, networkRepository);
    }
}

class ClientTest implements NetworkEventListener {
    NetworkRepository networkRepository;

    public static void main(String[] args) {
        new ClientTest();
    }

    public ClientTest() {
        super();
        networkRepository = new TCPNetworkRepository(false, this);
        var thread = new Thread((Runnable) networkRepository);
        thread.setDaemon(true);
        thread.start();

        Scanner scanner = new Scanner(System.in);
        String input;
        boolean running = true;

        while (running) {
            System.out.println("Please select an option:");
            System.out.println("1. CLOSE_CONNECTION");
            System.out.println("2. CLEAR_OBJECTS");
            System.out.println("3. REQUEST_OBJ_BY_INDEX");
            System.out.println("4. REQUEST_OBJ_LIST");
            System.out.println("5. REQUEST_OBJ_LIST_SIZE");
            System.out.println("0. Exit");

            input = scanner.nextLine();

            switch (input) {
                case "1" -> networkRepository.closeConnection();
                case "2" -> networkRepository.clearObjects();
                case "3" -> networkRepository.requestObjectByIndex(1);
                case "4" -> networkRepository.requestObjectsList();
                case "5" -> networkRepository.requestObjectsListSize();
                case "0" -> running = false;
                default -> System.out.println("Invalid input, please try again.");
            }
        }
        scanner.close();
    }

    @Override
    public void onEvent(NetworkEvent event) {
        System.out.println("EVENT!");
        Util.handleEvent(event, networkRepository);
    }
}
