package network;

import objects.GraphicalObject;
import utils.ObjectInfo;

public interface NetworkRepository {
    void closeConnection();

    void clearObjects();

    void sendObject(GraphicalObject object);

    void sendObjectByIndex(int index, GraphicalObject object);

    void sendObjectsList(ObjectInfo[] objects);

    void sendObjectsListSize(int size);

    void requestObjectByIndex(int index);

    void requestObjectsList();

    void requestObjectsListSize();
}
