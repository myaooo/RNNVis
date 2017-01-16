import cProfile, pstats, io


if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()
    # ... do something ...


    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    with open('profile.txt', 'w+') as f:
        f.write(s.getvalue())