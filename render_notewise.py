import music21

def render_notewise(score, output_filename, sample_freq = 12, note_offset = 0):
    speed = 1. / sample_freq
    piano_notes = []
    time_offset = 0

    for i,word in enumerate(score):
        if word.startswith('endp'):
            pass
        elif word.startswith('wait'):
            time_offset += int(word[4:])
        elif word.startswith('p'):
            duration = 1
            has_end = False
            for j in range(1, 200):
                if score[i + j][:4] == "wait":
                    duration += int(score[i + j][4:])
                if score[i + j][:3 + len(word)] == "end" + score[i] or score[i + j][:len(word)] == score[i]:
                    has_end = True
                    break

            if not has_end:
                duration = 12

            new_note = music21.note.Note(int(word[1:]) + note_offset)
            new_note.duration = music21.duration.Duration(duration * speed)
            new_note.offset = time_offset * speed
            piano_notes.append(new_note)

            # time_offset += add_wait

    piano = music21.instrument.fromString("Piano")
    piano_notes.insert(0, piano)
    piano_stream = music21.stream.Stream(piano_notes)
    main_stream = music21.stream.Stream([piano_stream])
    main_stream.write('midi', fp=output_filename)


def render_notewise_file(notewise_file, output_filename):
    with open(notewise_file, 'r') as f:
        render_notewise(f.read().split(' '), output_filename)