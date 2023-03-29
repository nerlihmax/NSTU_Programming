package network;

import objects.GraphicalObject;

public interface NetworkRepository {
    void sendObject(GraphicalObject object);

    void requestObjectByIndex(int index);

    void requestObjectsList();

    void requestObjectsListNames();

    void removeObjectByIndex(int index);
}
