package utils.network_events;

public sealed interface NetworkEvent permits CloseConnection,
        ClearObjects,
        RequestObjectByIndex,
        RequestObjectList,
        RequestObjectListSize,
        ResponseObjectByIndex,
        ResponseObjectList,
        ResponseObjectListSize,
        ResponseObject {
}