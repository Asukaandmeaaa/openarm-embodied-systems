"""Classified collection paths and append-only numbering; no robot imports."""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re


TAGS=('nearby','spread','rotation')


@dataclass(frozen=True)
class CollectionLayout:
    session: Path
    episodes: Path
    runs: Path
    classified: bool


def create_layout(base,tag='nearby',output=None,*,camera_only=False,stamp=None):
    if tag not in TAGS:raise ValueError('Unknown collection category: '+str(tag))
    base=Path(base).resolve()
    stamp=stamp or datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if output is not None:
        session=Path(output).resolve()
        layout=CollectionLayout(session,session/'episodes',session/'runs',False)
    elif camera_only:
        session=base/'runs/camera_checks'/('session_'+stamp)
        layout=CollectionLayout(session,session/'episodes',session/'runs',False)
    else:
        category=base/'collections'/tag
        session=category/'sessions'/('session_'+tag+'_'+stamp)
        layout=CollectionLayout(session,category/'episodes',category/'runs',True)
    # Never reinterpret an existing category/session as an overwrite target.
    layout.session.mkdir(parents=True,exist_ok=False)
    return layout


def next_episode_number(layout):
    """Account for failed/deleted episodes whose execution evidence remains."""
    numbers=[]
    for directory,prefix in ((layout.episodes,'episode'),(layout.runs,'cycle'),
                             (layout.runs.parent/'failed','episode')):
        if not directory.exists():continue
        for path in directory.iterdir():
            match=re.fullmatch(prefix+r'_(\d+)(?:_FAILED|_INTERRUPTED)?',path.name)
            if match:numbers.append(int(match[1]))
    return max(numbers,default=-1)+1


def mark_failed_attempt(layout, attempt, *, reason, interrupted=False):
    """Visible failure bucket, without moving files a child might still own.

    Even failures before the recorder starts get a marker. Original episode/run
    paths remain stable for logs, replay and recovery; no automatic retry.
    """
    from .collection_io import write_json
    label = 'INTERRUPTED' if interrupted else 'FAILED'
    marker_dir = layout.runs.parent/'failed'/f'episode_{attempt["number"]:03d}_{label}'
    marker_dir.mkdir(parents=True, exist_ok=False)
    marker = dict(number=attempt['number'], status=label.lower(), reason=reason,
                  episode=attempt['episode'], run=attempt['run'],
                  session=str(layout.session/'session.json'),
                  recording_started=(Path(attempt['episode'])/'metadata.json').is_file(),
                  hardware_state='unknown; inspect gateway and measured feedback before recovery',
                  eligible_for_training=False)
    marker_file = marker_dir/(label+'.json')
    write_json(marker_file, marker)
    for directory in (Path(attempt['episode']), Path(attempt['run'])):
        if directory.is_dir():
            write_json(directory/(label+'.json'), dict(marker, failure_marker=str(marker_file)))
    attempt.update(status=label.lower(), error=reason, failure_marker=str(marker_file))
    return marker_file


def archive_categories(base, tags=TAGS, *, execute=False, stamp=None):
    """Explicit dataset reset, never deletion or a hardware operation.

    Refuse while either collector or standalone cycle owns its workspace lock.
    Preserve each entire category outside collections, then create empty roots.
    Journal each move so a partially completed archive remains recoverable.
    """
    import fcntl
    from contextlib import ExitStack
    from .collection_io import write_json

    base = Path(base).resolve()
    tags = tuple(tags)
    if not tags or len(set(tags)) != len(tags) or any(tag not in TAGS for tag in tags):
        raise ValueError('Choose unique known collection categories')
    collections = base/'collections'
    archive_root = base/'collection_archives'
    if collections.is_symlink() or archive_root.is_symlink():
        raise ValueError('Refusing symlink collection/archive root')
    stamp = stamp or datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', stamp):
        raise ValueError('Unsafe archive stamp')
    archive = archive_root/('reset_'+stamp)
    with ExitStack() as stack:
        for name in ('.collection.lock', '.cycle.lock'):
            lock = stack.enter_context((base/name).open('a'))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('Collection/cycle is running; stop it safely before resetting storage') from exc
        items = []
        for tag in tags:
            source = collections/tag
            if source.is_symlink() or (source.exists() and not source.is_dir()):
                raise ValueError('Refusing unexpected category path: '+str(source))
            if source.exists():
                items.append(dict(category=tag, source=str(source), destination=str(archive/tag),
                    episodes=len(list((source/'episodes').glob('episode_*'))),
                    runs=len(list((source/'runs').glob('cycle_*'))), moved=False))
        report = dict(hardware_io_performed=False, deleted_files=0, archive=str(archive),
                      execute=execute, status='planned', items=items,
                      restore='Stop collection; move the new category aside, then move the archived category back. Never merge/overwrite recordings.')
        if not execute:
            return report
        if not items:
            raise ValueError('No existing collection categories to archive')
        archive.mkdir(parents=True, exist_ok=False)
        journal = archive/'archive_manifest.json'
        write_json(journal, report)
        try:
            for item in items:
                Path(item['source']).rename(item['destination'])
                item['moved'] = True
                write_json(journal, report)
                for child in ('episodes', 'runs', 'sessions'):
                    (Path(item['source'])/child).mkdir(parents=True, exist_ok=False)
            report['status'] = 'complete'
            write_json(journal, report)
        except BaseException as exc:
            report.update(status='incomplete', error=str(exc) or type(exc).__name__)
            write_json(journal, report)
            raise
        return report


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description='Archive old collection data, then restart numbering at zero. No robot operations.')
    parser.add_argument('--tag', choices=(*TAGS, 'all'), default='all')
    parser.add_argument('--archive-existing', action='store_true', help='Actually archive; otherwise only show the plan')
    args = parser.parse_args()
    print(json.dumps(archive_categories(Path(__file__).resolve().parent,
        TAGS if args.tag == 'all' else (args.tag,), execute=args.archive_existing), ensure_ascii=False, indent=2))
