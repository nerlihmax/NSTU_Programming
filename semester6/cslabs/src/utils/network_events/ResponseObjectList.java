package utils.network_events;

import objects.GraphicalObject;

public record ResponseObjectList(GraphicalObject[] objects) implements NetworkEvent {
}
