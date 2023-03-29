package utils.network_events;

public sealed interface NetworkEvent permits
        ResponseObjectList,
        ResponseObjectListNames,
        ResponseObject {
}