package utils.network_events;

public record ResponseObject(String object, String type) implements NetworkEvent {
}
