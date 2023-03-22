package utils.network_events;

public record ResponseObjectByIndex(int index, String objectType, String object) implements NetworkEvent {
}
