SELECT distinct author_id as id
FROM Views
WHERE author_id = viewer_id
order by author_id asc;