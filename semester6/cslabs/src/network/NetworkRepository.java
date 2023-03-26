package network;

import objects.GraphicalObject;

public interface NetworkRepository {
    void closeConnection();

    void clearObjects();

    void sendObject(GraphicalObject object);

    void sendObjectByIndex(int index, GraphicalObject object);

    void sendObjectsList(GraphicalObject[] objects);

    void sendObjectsListSize(int size);

    void requestObjectByIndex(int index);

    void requestObjectsList();

    void requestObjectsListSize();
}
