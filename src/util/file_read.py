def parse_file(file):
    all_stories = []
    story_file = open(file, "r")
    curr_story = []

    for line in story_file:
        if "###" in line:
            all_stories.append(curr_story)
            curr_story = []
        else:
            lp = line.split(" ")
            curr_story.append(" ".join(lp[1:]).strip())
    
    story_file.close()

    return all_stories
###

def parse_gold(file):
    all_stories = []
    story_file = open(file, "r")
    curr_story = []

    for line in story_file:
        if "###" in line:
            all_stories.append(curr_story)
            curr_story = []
        elif "@" in line:
            # cluster title, ignore
            pass
        else:
            lp = line.split(" ")
            curr_story.append(" ".join(lp[1:]).strip())

    story_file.close()

    return all_stories
###