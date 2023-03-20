package network;

import objects.GraphicalObject;
import utils.ObjectInfo;

import java.io.InputStream;
import java.util.List;

public interface NetworkRepository {
    void closeConnection();

    void clearObjects();

    void sendObject(GraphicalObject object);

    void sendObjectsList(List<ObjectInfo> objects);

    void sendObjectsListSize(int size);

    InputStream getInputStream();
}
