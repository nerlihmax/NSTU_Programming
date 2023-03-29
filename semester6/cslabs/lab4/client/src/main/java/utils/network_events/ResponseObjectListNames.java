package utils.network_events;

import objects.GraphicalObject;

public record ResponseObjectListNames(GraphicalObject[] objects) implements NetworkEvent {
}