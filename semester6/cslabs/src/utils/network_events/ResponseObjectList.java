package utils.network_events;

import utils.ObjectInfo;

public record ResponseObjectList(ObjectInfo[] objects) implements NetworkEvent {
}
