package utils.network_events;

import objects.GraphicalObject;

public record ResponseObject(GraphicalObject object) implements NetworkEvent {
}
