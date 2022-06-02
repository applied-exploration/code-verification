def strip_comments(source: str) -> str:
    import pyparsing

    # commentFilter = pyparsing.cppStyleComment.suppress()
    # To filter python style comment, use
    commentFilter = pyparsing.pythonStyleComment.suppress()
    # To filter C style comment, use
    # commentFilter = pyparsing.cStyleComment.suppress()

    return commentFilter.transformString(source)
