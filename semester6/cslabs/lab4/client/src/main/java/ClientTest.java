import network.NetworkEventListener;
import network.NetworkRepository;
import network.RESTNetworkRepository;
import network.TCPNetworkRepository;
import objects.GraphicalObject;
import objects.Smiley;
import objects.Star;
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
                networkRepository.sendObjectsList(new GraphicalObject[]{new Star(10, 10, 10, 10, Color.RED), new Smiley(10, 10, 10, 10, Color.black)});
            }
            case ClearObjects ignored -> System.out.println("ClearObjects");
            case ResponseObjectByIndex responseObjectByIndex -> {
                System.out.println("ResponseObjectByIndex: " + "index: " + responseObjectByIndex.index() + " object: " + responseObjectByIndex.object());
                var object = switch (responseObjectByIndex.type()) {
                    case "Star" -> new Star(100, 100, 100, 100, Color.RED);
                    case "Smiley" -> new Smiley(100, 100, 100, 100, Color.RED);
                    default -> throw new IllegalStateException("Unexpected value: " + responseObjectByIndex.type());
                };
                object.readFromJson(responseObjectByIndex.object());
                System.out.println(object.writeToJson());
            }
            case CloseConnection ignored -> System.out.println("CloseConnection");
            case RequestObjectListSize ignored -> {
                System.out.println("RequestObjectListSize");
                networkRepository.sendObjectsListSize(2);
            }
            case ResponseObjectList responseObjectList ->
                    System.out.println("ResponseObjectList: " + Arrays.toString(responseObjectList.objects()));
            case ResponseObjectListSize responseObjectListSize ->
                    System.out.println("ResponseObjectListSize: " + responseObjectListSize.size());
            case ResponseObject responseObject -> System.out.println("ResponseObject: " + responseObject.object());
        }
    }
}


public class ClientTest implements NetworkEventListener {
    NetworkRepository networkRepository;

    public static void main(String[] args) {
        new ClientTest();
    }

    public ClientTest() {
        super();
        networkRepository = new RESTNetworkRepository(this);

        Scanner scanner = new Scanner(System.in);
        String input;
        boolean running = true;

        while (running) {
            System.out.println("Please select an option:");
            System.out.println("1. REQUEST_OBJ_BY_INDEX");
            System.out.println("2. REQUEST_OBJ_LIST");
            System.out.println("3. REQUEST_OBJ_LIST_SIZE");
            System.out.println("0. Exit");

            input = scanner.nextLine();

            switch (input) {
                case "1" -> networkRepository.requestObjectByIndex(1);
                case "2" -> networkRepository.requestObjectsList();
                case "3" -> networkRepository.requestObjectsListSize();
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
