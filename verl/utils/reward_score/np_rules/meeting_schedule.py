from collections import defaultdict

from .common import maximization_score, parse_answer_literal


def validation(data, answer, ground_truth):
    """Validate a meeting-schedule solution."""
    if not isinstance(data, dict):
        return True, -1, "input must be a dictionary"

    try:
        schedule = parse_answer_literal(answer)
    except (ValueError, SyntaxError) as exc:
        return True, -1, str(exc)

    if not isinstance(schedule, list):
        return True, -1, "answer must be a list of schedule entries"

    meetings = data.get("meetings", {})
    availability = data.get("attendee_availability", {})
    rooms = data.get("rooms", {})
    if not isinstance(meetings, dict) or not isinstance(availability, dict) or not isinstance(rooms, dict):
        return True, -1, "input must contain meetings, attendee_availability, and rooms dictionaries"

    room_schedule = defaultdict(list)
    attendee_schedule = defaultdict(list)
    seen_meetings = set()

    for entry in schedule:
        if not (isinstance(entry, (tuple, list)) and len(entry) == 3):
            return True, -1, f"invalid entry: {entry!r}, expected (meeting_id, room_id, start_time)"

        meeting_id, room_id, start = entry
        meeting_key = str(meeting_id)
        room_key = str(room_id)

        if meeting_key not in meetings:
            return True, -1, f"invalid meeting ID: {meeting_id}"
        if room_key not in rooms:
            return True, -1, f"invalid room ID: {room_id}"
        if meeting_key in seen_meetings:
            return True, -1, f"meeting {meeting_id} scheduled more than once"
        seen_meetings.add(meeting_key)

        meeting_info = meetings[meeting_key]
        duration = meeting_info.get("duration")
        attendees = meeting_info.get("attendees", [])
        if not isinstance(duration, int) or duration <= 0:
            return True, -1, f"invalid duration for meeting {meeting_id}"

        try:
            start = int(start)
        except (TypeError, ValueError):
            return True, -1, f"invalid start time for meeting {meeting_id}"
        end = start + duration

        room_info = rooms[room_key]
        capacity = room_info.get("capacity") if isinstance(room_info, dict) else room_info
        if not isinstance(capacity, int) or len(attendees) > capacity:
            return True, -1, f"meeting {meeting_id} exceeds capacity of room {room_id}"

        for previous_meeting, prev_start, prev_end in room_schedule[room_key]:
            if start < prev_end and prev_start < end:
                return True, -1, f"meeting {meeting_id} overlaps with meeting {previous_meeting} in room {room_id}"
        room_schedule[room_key].append((meeting_key, start, end))

        for attendee in attendees:
            attendee_key = str(attendee)
            if attendee_key not in availability:
                return True, -1, f"attendee {attendee} availability not found"
            for previous_meeting, prev_start, prev_end in attendee_schedule[attendee_key]:
                if start < prev_end and prev_start < end:
                    return True, -1, (
                        f"attendee {attendee} is double-booked in meetings {previous_meeting} and {meeting_id}"
                    )
            is_available = any(
                start >= available_start and end <= available_end
                for available_start, available_end in availability[attendee_key]
            )
            if not is_available:
                return True, -1, f"attendee {attendee} is not available for meeting {meeting_id} at {start}-{end}"
            attendee_schedule[attendee_key].append((meeting_key, start, end))

    scheduled_meetings = len(schedule)
    score = maximization_score(scheduled_meetings, ground_truth)
    return False, score, f"scheduled {scheduled_meetings} meetings, ground truth: {ground_truth}"
