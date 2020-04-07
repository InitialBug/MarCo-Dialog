

examples = [
    {
        "file": "FILENAME",
        "info": [
            {
                "turn_num": 1,
                "user": "USER QUERY",
                "system": "HUMAN RESPONSE",
                "HDSA": "HDSA RESPONSE",
                "MarCo": "MarCo RESPONSE",
                "MarCo vs. system":
                    {
                        "Readability":
                            ["Tie", "MarCo", "System"],
                        "Completion":
                            ["MarCo", "MarCo", "Tie"]
                    }
            },
            ...
        ]
    }

]