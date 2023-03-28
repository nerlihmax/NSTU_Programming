package utils.network_events;

public record ResponseObjectByIndex(int index, String type, String object) implements NetworkEvent {
}
